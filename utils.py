import os
import matplotlib.pyplot as plt


def save_plot(fig, filename, folder="results"):
    """
    Saves a matplotlib figure to the results folder.
    - fig: matplotlib figure object (plt.gcf() or returned fig)
    - filename: string (e.g., "cnn_training_curves.png")
    - folder: output folder (default: "results")
    """
    os.makedirs(folder, exist_ok=True)  # create folder if not exists
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"[INFO] Saved plot to {path}")
    plt.close(fig)
