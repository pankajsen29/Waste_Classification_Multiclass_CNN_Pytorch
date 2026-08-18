####################################################################
# Step 0: defines the configurations.
####################################################################

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# data preprocessing setting
DATA_DIR = "data/RealWaste"
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # ImageNet normalization (mandatory for pretrained models)
IMAGENET_STD  = [0.229, 0.224, 0.225]   # ImageNet normalization (mandatory for pretrained models)

# model settings for used for training, tuining, test/prediction
MODEL_NAME = "resnet18"             # primary - main CNN result
# MODEL_NAME = "resnet34"           # baseline
# MODEL_NAME = "efficientnet_b0"    # best final model
OPTIMIZER_NAME = "adam"
# OPTIMIZER_NAME = "sgd"
LEARNING_RATE = 0.001
# LEARNING_RATE = 0.01              # only with efficientnet_b0

# settings for model state saving
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CHECKPOINT_FILE = CHECKPOINTS_DIR / "resnet18_crossentropyloss_adam_lr_0.001_epoch_10_batch_32_model_state.pth"
TRAINING_HISTORY_JSON = CHECKPOINTS_DIR / "resnet18_crossentropyloss_adam_lr_0.001_epoch_10_batch_32_train_history.json"
# MODEL_CHECKPOINT_FILE = CHECKPOINTS_DIR / "resnet34_crossentropyloss_adam_lr_0.001_epoch_10_batch_32_model_state.pth"
# TRAINING_HISTORY_JSON = CHECKPOINTS_DIR / "resnet34_crossentropyloss_adam_lr_0.001_epoch_10_batch_32_train_history.json"
# MODEL_CHECKPOINT_FILE = CHECKPOINTS_DIR / "efficientnet_b0_crossentropyloss_sgd_lr_0.001_epoch_10_batch_32_model_state.pth"
# TRAINING_HISTORY_JSON = CHECKPOINTS_DIR / "efficientnet_b0_crossentropyloss_sgd_lr_0.001_epoch_10_batch_32_train_history.json"

