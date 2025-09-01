import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import save_plot


# ---------------- Dataset ----------------
def load_data(batch_size=64, dataset="MNIST"):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset.lower() == "mnist":
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


# ---------------- Model ----------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# ---------------- Training ----------------
def train_model(model, trainloader, testloader, epochs=5, lr=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_accuracies = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        train_losses.append(avg_loss)
        acc = evaluate(model, testloader, device)
        test_accuracies.append(acc)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}")
    return train_losses, test_accuracies


# ---------------- Evaluation ----------------
def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    save_plot(fig, "cnn_confusion_matrix.png")
    return acc


# ---------------- Run ----------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainloader, testloader = load_data(dataset="FashionMNIST")
    model = CNN(num_classes=10).to(device)
    losses, accs = train_model(model, trainloader, testloader, epochs=5, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses, label="Training Loss")
    axes[0].legend()
    axes[1].plot(accs, label="Test Accuracy")
    axes[1].legend()
    save_plot(fig, "cnn_training_curves.png")
