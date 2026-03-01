#########################################################################################
#Step 6: Model evaluation on test set (accuracy, precision, recall, F1, confusion matrix)
#########################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

#Collect predictions on test set: No gradients, No weight updates, Pure evaluation
def get_all_predictions(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

#Compute evaluation metrics, average="weighted" : Handles class imbalance and recommended for multi-class
def compute_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted")
    }
    return metrics

#Confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, class_names):
    
    # cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm,
    #     display_labels=class_names
    # )
    # disp.plot(cmap="Blues", xticks_rotation=45)

    #todo: need to fix the alignments of the predicted labels in the matrix
    ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=class_names,
    cmap="Greens",
    xticks_rotation=45)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

#Full evaluation pipeline
def evaluate_model(model, test_loader, device, class_names):
    y_true, y_pred = get_all_predictions(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)

    print("\nTest Set Performance:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    plot_confusion_matrix(y_true, y_pred, class_names)

    return metrics
