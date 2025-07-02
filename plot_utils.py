import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_history(histories, save_path, metrics):
    """
    Plot training and validation metrics in one figure per metric with subplots for each phase.

    Args:
        histories (dict): Dictionary containing histories per training phase.
        save_path (str): Directory path to save plots.
        metrics (list): List of metric names to plot.
    """
    os.makedirs(save_path, exist_ok=True)

    phases = ["head", "fine_1", "fine_2", "fine_3"]

    for metric in metrics:
        plt.figure(figsize=(16, 10))
        plotted_any = False

        for i, phase_name in enumerate(phases):
            hist = histories.get(phase_name, {})
            if not hist:
                continue

            history_dict = hist.history if hasattr(hist, "history") else hist

            train_metric = history_dict.get(metric, [])
            val_metric = history_dict.get(f"val_{metric}", [])

            if train_metric or val_metric:
                plotted_any = True
                plt.subplot(2, 2, i+1)
                if train_metric:
                    plt.plot(range(len(train_metric)), train_metric, label="train")
                if val_metric:
                    plt.plot(range(len(val_metric)), val_metric, label="val")
                plt.title(f"{phase_name.capitalize()} {metric.capitalize()}")
                plt.xlabel("Epoch")
                plt.ylabel(metric.capitalize())
                plt.legend()
                plt.grid(True)

        if plotted_any:
            plt.tight_layout()
            metric_filename = f"{metric}_curve.png"
            metric_save_path = os.path.join(save_path, metric_filename)
            plt.savefig(metric_save_path)
            print(f"[DEBUG] Saved {metric} plot with all phases to {metric_save_path}")
        else:
            print(f"[DEBUG] No data available for metric: {metric}")

        plt.close()




def save_confusion_matrix(y_true, y_pred, labels, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_roc_curve(y_true, y_scores, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
