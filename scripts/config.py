from pathlib import Path

# project folders
base_dir = Path(__file__).resolve().parents[1]

data_dir = base_dir / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"

mitbih_dir = raw_dir / "mitbih"
ptbxl_dir = raw_dir / "ptbxl"

model1_dir = processed_dir / "model1"
model2_dir = processed_dir / "model2"


# settings for model 1
# MIT-BIH is sampled at 360 Hz, so a 2-second window has 720 samples
mitbih_fs = 360
mitbih_lead_index = 0

window_sec = 2.0
stride_sec = 1.0


# settings for model 2
# PTB-XL records100 is sampled at 100 Hz
ptbxl_fs = 100
ptbxl_lead_index = 0

# each beat segment has 25 samples before the peak
# and 45 samples after the peak
beat_before_sec = 0.25
beat_after_sec = 0.45


# broad diagnostic classes used for model 2
target_superclasses = ["NORM", "MI", "HYP", "CD", "STTC"]

class_name_map = {
    "NORM": "Normal",
    "MI": "Myocardial_Infarction",
    "HYP": "Hypertrophy",
    "CD": "Conduction_Disorder",
    "STTC": "ST_T_Change",
}