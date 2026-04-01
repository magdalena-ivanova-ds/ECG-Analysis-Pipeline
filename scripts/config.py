from pathlib import Path

# Base folders
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MITBIH_DIR = RAW_DIR / "mitbih"
PTBXL_DIR = RAW_DIR / "ptbxl"

MODEL1_DIR = PROCESSED_DIR / "model1"
MODEL2_DIR = PROCESSED_DIR / "model2"

# MIT-BIH settings
MITBIH_FS = 360
MITBIH_LEAD_INDEX = 0
WINDOW_SEC = 2.0
STRIDE_SEC = 1.0

# PTB-XL settings
PTBXL_FS = 100
PTBXL_LEAD_INDEX = 0
BEAT_BEFORE_SEC = 0.25
BEAT_AFTER_SEC = 0.45

# Keep labels simple and broad
TARGET_SUPERCLASSES = ["NORM", "MI", "HYP", "CD", "STTC"]

CLASS_NAME_MAP = {
    "NORM": "Normal",
    "MI": "Myocardial_Infarction",
    "HYP": "Hypertrophy",
    "CD": "Conduction_Disorder",
    "STTC": "ST_T_Change"
}