import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

def plot_history(histories, save_path, metrics):
    """
    Plot training and validation metrics for all training phases.

    Args:
        histories (dict): Dictionary containing histories per training phase.
        save_path (str): Directory path to save plots.
        metrics (list): List of metric names to plot.
    """
    os.makedirs(save_path, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plotted_any = False

        # Loop through each phase history: head, fine, fine_2
        for phase_name, hist in histories.items():
            if not hist:
                continue

            # Determine if hist is a History object or dict
            history_dict = hist.history if hasattr(hist, "history") else hist

            train_metric = history_dict.get(metric, [])
            val_metric = history_dict.get(f"val_{metric}", [])

            # Plot training metric
            if train_metric:
                plt.plot(range(len(train_metric)), train_metric, label=f"{phase_name} - train")
                plotted_any = True

            # Plot validation metric
            if val_metric:
                plt.plot(range(len(val_metric)), val_metric, label=f"{phase_name} - val")
                plotted_any = True

        if plotted_any:
            plt.title(f"{metric.capitalize()} over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)

            # Save plot
            metric_filename = f"{metric}_curve.png"
            metric_save_path = os.path.join(save_path, metric_filename)
            plt.savefig(metric_save_path)
            print(f"[DEBUG] Saved {metric} plot to {metric_save_path}")
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
