import src.config as cfg
DEBUG = True

if DEBUG:
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    #NUM_EPOCHS = 10
    #BATCH_SIZE = 16
    #NUM_WORKERS = 0

else:
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    # NUM_EPOCHS = 20
    # BATCH_SIZE = 16
    # NUM_WORKERS = 2


######## dataset ######
from src.dataset import get_dataloaders

train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

#TESTCODE
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)
#o/p: torch.Size([8, 3, 224, 224]) torch.Size([8])
#images=a single batch pulled from train_loader
#lables=targets for that batch


######## model #########
from src.model import (
    get_model,
    get_loss_function,
    get_optimizer,
    get_device
)

device = get_device()
model = get_model(cfg.MODEL_NAME, num_classes)
model = model.to(device)

#note-loss and optimizer are useless unless training
criterion = get_loss_function()
optimizer = get_optimizer(model, optimizer_name=cfg.OPTIMIZER_NAME, lr=cfg.LEARNING_RATE)

######### train #######
import json
from src.train import (
    train_model,
    load_model_checkpoint,
    dummy_training1, 
    dummy_training2,
    test_one_training_step
)

#TESTCODE
#dummy_training1(model, images)

#TESTCODE
#ensure optimizer, criterion are set
#dummy_training2(model, labels, device, optimizer, criterion)

#TESTCODE
#test_one_training_step(model, images, labels, optimizer, criterion)

if DEBUG:
    # 1. load checkpoint first
    checkpoint = load_model_checkpoint(cfg.MODEL_CHECKPOINT_FILE, device)

    # 2. load trained weights
    model.load_state_dict(checkpoint["model"])
    class_names = checkpoint["class_names"]
    optimizer = checkpoint["optimizer"]
    
    # 3. loading the history from json
    with open(cfg.TRAINING_HISTORY_JSON, "r") as f:
        history = json.load(f)

else:
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        class_names,
        device,
        cfg.MODEL_CHECKPOINT_FILE,
        num_epochs=NUM_EPOCHS
    )

    #saving trained model weights is done during training based on best validation loss

    #saving the history to json    
    with open(cfg.TRAINING_HISTORY_JSON, "w") as f:
        json.dump(history, f)

    #loading the history from json
    with open(cfg.TRAINING_HISTORY_JSON, "r") as f:
        history = json.load(f)


'''
######### plots ##############
from src.plots import plot_learning_curves

plot_learning_curves(history)

#how to interpret learning curves
#Train (Down), Val (Down) : Good fit
#Train (Down), Val (Up) :  Overfitting
#Both high & flat : Underfitting
'''

'''
############## tuning ###################
from src.tuning import hyperparameter_search

results = hyperparameter_search(
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    class_names=class_names,
    learning_rates=[1e-4, 1e-3, 1e-2, 1e-1], #[0.0001, 0.001, 0.01, 0.1]
    batch_sizes=[16, 32],
    num_epochs=5
)

print(results)

#What hyperparameters are tuned
#Learning rate: how fast the model learns
#Batch size: stability vs speed trade-off
#Epochs: training duration
'''


############# evaluate #################
from src.evaluate import evaluate_model

test_metrics = evaluate_model(
    model,
    test_loader,
    device,
    class_names
)