
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
model = get_model("resnet18", num_classes) #primary - main CNN result
#model = get_model("resnet34", num_classes) #baseline
#model = get_model("efficientnet_b0", num_classes) #best final model
model = model.to(device)

#note-loss and optimizer are useless unless training
criterion = get_loss_function()
optimizer = get_optimizer(model, optimizer_name="adam", lr=0.001)
#optimizer = get_optimizer(model, optimizer_name="sgd", lr=0.001)


######### train #######
from src.train import (
    train_model, 
    save_trained_model,
    load_trained_model,
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


#for saving the model state
from pathlib import Path
save_dir = Path("checkpoints")
save_dir.mkdir(parents=True, exist_ok=True)

if DEBUG:
    #existing model needs to passed, to load the saved weights from disc
    model, history = load_trained_model(model, save_dir / "waste_seg_full_training_state.pth")

else:
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS
    )

    #saving trained model weights
    save_trained_model(model, optimizer, history, save_dir / "waste_seg_full_training_state.pth")

    #saving the history to json
    import json    
    with open(save_dir / "waste_seg_history.json", "w") as f:
        json.dump(history, f)

    #loading the history from json
    with open(save_dir / "waste_seg_history.json", "r") as f:
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
    num_classes=num_classes,
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
