import datetime

class Config:
    RUN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # COCO paths
    COCO_IMG_TRAIN = "./data/coco/images/train2017/"
    COCO_IMG_VAL   = "./data/coco/images/val2017/"
    COCO_ANN_TRAIN = "./data/coco/annotations/instances_train2017.json"
    COCO_ANN_VAL   = "./data/coco/annotations/instances_val2017.json"

    # Training params
    BATCH_SIZE    = 4
    LR            = 1e-5
    WEIGHT_DECAY  = 1e-4
    EPOCHS        = 50
    NUM_WORKERS   = 4

    # Loss weights
    CLASS_WEIGHT  = 1.0
    BBOX_WEIGHT   = 5.0
    GIOU_WEIGHT   = 2.0

    # Model parameters
    # Typical COCO usage => 80 classes + background => 81
    NUM_CLASSES        = 81
    NUM_QUERIES        = 100
    D_MODEL            = 256
    NHEAD              = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD    = 1024
    DROPOUT            = 0.1
    NUM_FEATURE_LEVELS = 4
    ENC_N_POINTS       = 4
    DEC_N_POINTS       = 4

    # Additional
    AUX_LOSS       = True  # Whether to use auxiliary losses from intermediate decoder layers
    USE_NMS        = True  # Whether to apply NMS after detection
    SCORE_THRESH   = 0.5   # Confidence threshold for postprocess
    NMS_THRESH     = 0.5   # IoU threshold for NMS

    OUTPUT_DIR = f"./output/rt_detr_v2_{RUN_DATE}"
