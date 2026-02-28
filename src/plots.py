#########################################
#Step 4: Learning-Curve Plots
#########################################

import matplotlib.pyplot as plt


def plot_learning_curves(history):
    """
    history: dict with keys
    - train_loss
    - val_loss
    - train_acc
    - val_acc
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    # ---- Loss curve ----
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Accuracy curve ----
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()
