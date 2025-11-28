import os 

SEED = int(os.getenv("SEED", "42"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE","32"))
NUM_WORKERS = 2
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
VAL_PATIENCE = 5
WARMUP_EPOCHS = 5

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18") 
