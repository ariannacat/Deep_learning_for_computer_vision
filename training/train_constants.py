import os 

SEED = int(os.getenv("SEED", "42"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE","32"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS","2"))
LR = float(os.getenv("LR", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
EPOCHS = int(os.getenv("EPOCHS", "30"))
VAL_PATIENCE = int(os.getenv("VAL_PATIENCE", "5"))
WARMUP_EPOCHS = int(os.getenv("WARMUP_EPOCHS", "5"))

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "resnet18") 
