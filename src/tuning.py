###############################
#Step 5: Hyperparameter Tuning
###############################

from src.model import get_model, get_loss_function, get_optimizer
from src.train import train_model
import src.config as cfg

def hyperparameter_search(
    train_loader,
    val_loader,
    device,
    class_names,
    learning_rates,
    batch_sizes,
    num_epochs=5
):
    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTesting lr={lr}, batch_size={batch_size}")

            model = get_model(cfg.MODEL_NAME, len(class_names))
            criterion = get_loss_function()
            optimizer = get_optimizer(model, optimizer_name=cfg.OPTIMIZER_NAME, lr=lr)

            model, history = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                class_names,
                device,
                None,  # No model saving during hyperparameter tuning
                num_epochs=num_epochs
            )

            best_val_acc = max(history["val_acc"])

            results.append({
                "lr": lr,
                "batch_size": batch_size,
                "best_val_acc": best_val_acc
            })

            print(f"Best Val Accuracy: {best_val_acc:.4f}")

    return results