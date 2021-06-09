DATAPATH = "data"  # None to load dataset
BATCH_SIZE = 16  # BO -16

MODEL_KIND = "EfficientNetB0"
LEARNING_RATE = 1e-4
EMBEDDING_SIZE = 256
EPOCHS = 10
MARGIN = 0.01
MODEL_PATH = "models/b0_256_0.01"
AUGMENT = False

IMG_SIZE_DICT = {
    "EfficientNetB0": 224,
    "EfficientNetB1": 240,
    "EfficientNetB2": 260,
    "EfficientNetB3": 300,
    "EfficientNetB4": 380,
    "EfficientNetB5": 456,
    "EfficientNetB6": 528,
    "EfficientNetB7": 600,
}
IMG_SIZE = IMG_SIZE_DICT[MODEL_KIND]

# Evaluation
BATCH_SIZE_EVAL = 16  # 32 for EffNetB0 models
MODEL_PATH_PREDICT = "model_b3_256_m1"
k_max = 5
