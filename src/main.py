
DEBUG = True

if DEBUG:
    # NUM_EPOCHS = 2
    # BATCH_SIZE = 8
    # NUM_WORKERS = 0

    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    NUM_WORKERS = 0
else:
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    NUM_WORKERS = 2



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
    get_resnet18,
    get_loss_function,
    get_optimizer,
    get_device
)

device = get_device()
model = get_resnet18(num_classes)
model = model.to(device)

#note-loss and optimizer are useless unless training
criterion = get_loss_function()
optimizer = get_optimizer(model)

######### train #######
from src.train import (
    train_model,
    dummy_training1, 
    dummy_training2,
    test_one_training_step
)
#NUM_EPOCHS = 20

#TESTCODE
#dummy_training1(model, images)

#TESTCODE
#ensure optimizer, criterion are set
#dummy_training2(model, labels, device, optimizer, criterion)

#TESTCODE
#test_one_training_step(model, images, labels, optimizer, criterion)

model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=NUM_EPOCHS
    )