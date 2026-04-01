from pathlib import Path
import zipfile
import requests

from config import RAW_DIR, PTBXL_DIR
from utils_ecg import make_dirs

PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"


def download_file(url, save_path):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def download_ptbxl():
    make_dirs([RAW_DIR, PTBXL_DIR])

    zip_path = RAW_DIR / "ptbxl.zip"

    # strong existence check
    csv_exists = (PTBXL_DIR / "ptbxl_database.csv").exists()
    scp_exists = (PTBXL_DIR / "scp_statements.csv").exists()
    records_exists = (PTBXL_DIR / "records100").exists()

    if csv_exists and scp_exists and records_exists:
        print("PTB-XL already exists. Skipping download.")
        return

    if not zip_path.exists():
        print("Downloading PTB-XL...")
        download_file(PTBXL_URL, zip_path)
        print("PTB-XL zip download finished.")
    else:
        print("PTB-XL zip already downloaded.")

    print("Extracting PTB-XL...")
    extract_zip(zip_path, PTBXL_DIR)
    print("PTB-XL extraction finished.")