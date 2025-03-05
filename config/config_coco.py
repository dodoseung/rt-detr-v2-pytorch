import datetime

class Config:
    RUN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # COCO 이미지/어노테이션 경로
    # Download dataset from here: https://cocodataset.org/#download
    COCO_IMG_TRAIN = "./data/coco/images/train2017/"
    COCO_IMG_VAL = "./data/coco/images/val2017/"
    COCO_ANN_TRAIN = "./data/coco/annotations/instances_train2017.json"
    COCO_ANN_VAL   = "./data/coco/annotations/instances_val2017.json"

    # 학습 파라미터
    BATCH_SIZE = 16
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    NUM_WORKERS = 8

    # COCO 클래스 수 (80) + background 1 = 81로 가정
    NUM_CLASSES = 81   # 배경=0 + COCO main 80 classes
    NUM_QUERIES = 100
    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    NUM_FEATURE_LEVELS = 3
    ENC_N_POINTS = 4
    DEC_N_POINTS = 4

    OUTPUT_DIR = f"./outputs/coco_output_{RUN_DATE}"