"""
DermaScan AI — Central Configuration
 
All paths, hyperparameters, and constants live here.
No more hardcoded values scattered across files.
 
Usage:
    from configs.config import cfg
    print(cfg.DATA_PATH)
    print(cfg.BATCH_SIZE)
"""
 

import os 
from dataclasses import dataclass ,field
from typing import List


@dataclass
class Config:

    #Data Path
    DATA_PATH: str = "data/processed/full_dataset"
    CSV_PATH: str = "data/processed/full_metadata.csv"
    RAW_DATA_PATH: str = "data/raw/all_data"
    GT_CSV_PATH: str = "data/raw/ISIC_2019_Training_GroundTruth.csv"
    META_CSV_PATH: str = "data/raw/ISIC_2019_Training_Metadata.csv"

    #Models
    CHECKPOINT_DIR: str = "models/vision/best_model.ckpt"
    YOLO_PATH: str = "models/detector/best.pt"
    XGB_MODEL_PATH: str = "models/xgb_meta_learner.json"
    XGB_FEATURE_PATH: str = "models/xgb_features.pkl"

    #Model
    BACKBONE: str = "efficientnet_b3"
    NUM_CLASSES: int = 8
    IMAGE_SIZE: int = 300
    CLASSES: List[str] = field(default_factory=lambda: [
        '0_mel', '1_nv', '2_bcc', '3_ak',
        '4_bkl', '5_df', '6_vasc', '7_scc'
    ])


    #Training
    BATCH_SIZE: int = 16
    EPOCHS: int = 25
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY : float = 1e-4
    K_FOLDS: int = 5
    NUM_WORKERS: int = 4
    ACCUMULATE_GRAD_BATCHES: int = 4
    EARLY_STOP_PATIENCE: int = 7
    LABEL_SMOOTHING: float = 0.1
    FOCAL_GAMMA:float = 2.0
    RANDOM_SEED: int = 42

    #Fine-Tuning Mode Settings
    MODE: str = "finetune"
    DERM12345_TRAIN: str = "data/derm12345/train"
    DERM12345_VAL: str = "data/derm12345/val"
    FINE_TUNE_CHECKPOINT: str = "models/vision/midas_model.ckpt"
    FINTETUNE_LR: float = 1e-5 #Checkpoint is already in a good position high LR weights will ruin it.
    FINETUNE_BATCH: int = 8
    FINE_TUNE_ACCUM: int = 8
    FINETUNE_EPOCHS: int = 15


    #EVALUATION
    MEL_THRESHOLD: float = 0.11
    TTA_ENABLED: bool = True
    MEL_CLASS_IDX: int = 0


    #INFERENCE
    YOLO_CONF: float = 0.25
    YOLO_MAX_BOX_RATIO: float = 0.6
    CENTER_CROP_MARGIN: float = 0.20
    CNN_VISUAL_ALERT_THRESHOLD: float = 0.40


    #NORMALIZATION
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    #MALIGNANT CLASSES
    MALIGNANT_INDICES: List[int] = field(default_factory=lambda: [0, 2, 3, 7])
    BENIGN_INDICES: List[int] = field(default_factory=lambda: [1, 4, 5, 6])

cfg = Config()
