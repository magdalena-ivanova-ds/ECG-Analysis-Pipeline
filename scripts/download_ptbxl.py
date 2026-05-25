import zipfile
import requests
from config import raw_dir, ptbxl_dir
from utils_ecg import make_dirs


ptbxl_url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"


def download_file(url, save_path):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(extract_to)


def download_ptbxl():
    make_dirs([raw_dir, ptbxl_dir])

    zip_path = raw_dir / "ptbxl.zip"

    # check whether the important PTB-XL files are already available
    database_exists = (ptbxl_dir / "ptbxl_database.csv").exists()
    scp_file_exists = (ptbxl_dir / "scp_statements.csv").exists()
    records_folder_exists = (ptbxl_dir / "records100").exists()

    if database_exists and scp_file_exists and records_folder_exists:
        print("Ptb-xl data is already available, skipping download.")
        return

    if not zip_path.exists():
        print("Downloading ptb-xl data...")
        download_file(ptbxl_url, zip_path)
        print("Ptb-xl zip file downloaded.")
    else:
        print("Ptb-xl zip file already exists.")

    print("Extracting ptb-xl data...")
    extract_zip(zip_path, ptbxl_dir)
    print("Ptb-xl extraction completed.")