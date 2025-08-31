import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import save_plot


# ---------------- Tokenizer & Vocab ----------------
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def build_vocab(train_iter, max_size=20000):
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"], max_tokens=max_size)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


# ---------------- Dataset Preprocessing ----------------
def preprocess_data(train_iter, test_iter, vocab, max_len=200):
    def encode(text):
        tokens = tokenizer(text)
        ids = [vocab[token] for token in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def process_batch(batch):
        texts, labels = zip(*batch)
        encoded = [encode(t)[:max_len] for t in texts]  # truncate
        padded = pad_sequence(encoded, batch_first=True, padding_value=vocab["<pad>"])
        labels = torch.tensor([1 if l == "pos" else 0 for l in labels], dtype=torch.long)
        return padded, labels

    return process_batch


# ---------------- Model ----------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)  # take final hidden state
        x = self.relu(h_n[-1])
        x = self.fc(x)
        return x


# ---------------- Training ----------------
def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        acc = evaluate(model, test_loader, device)
        test_accuracies.append(acc)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}")

    return train_losses, test_accuracies

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(losses, label="Training Loss")
axes[0].legend()
axes[1].plot(accs, label="Test Accuracy")
axes[1].legend()
save_plot(fig, "lstm_training_curves.png")

# ---------------- Evaluation ----------------
def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=ax)
    save_plot(fig, "lstm_confusion_matrix.png")


# ---------------- Run ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_iter, test_iter = IMDB(split=("train", "test"))
    vocab = build_vocab(train_iter)

    # Re-initialize iterators (they get consumed)
    train_iter, test_iter = IMDB(split=("train", "test"))
    collate_fn = preprocess_data(train_iter, test_iter, vocab)

    # Dataloaders
    from torch.utils.data import DataLoader
    train_iter, test_iter = IMDB(split=("train", "test"))
    train_loader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(list(test_iter), batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    losses, accs = train_model(model, train_loader, test_loader, epochs=3, device=device)

    # Plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(losses, label="Training Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(accs, label="Test Accuracy")
    plt.legend()
    plt.show()
