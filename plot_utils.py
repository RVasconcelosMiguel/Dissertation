import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_history(histories, metrics=["loss", "accuracy"]):
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for name, hist in histories.items():
            if metric in hist.history:
                plt.plot(hist.history[metric], label=f"{name} - train")
                plt.plot(hist.history.get(f"val_{metric}", []), label=f"{name} - val")
        plt.title(f"{metric.capitalize()} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"logs/{metric}_curve.png")
        plt.close()

def save_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
