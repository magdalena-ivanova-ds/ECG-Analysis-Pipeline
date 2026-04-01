import wfdb

from config import MITBIH_DIR
from utils_ecg import make_dirs


def download_mitbih():
    make_dirs([MITBIH_DIR])

    # strong existence check
    needed_files = [
        MITBIH_DIR / "100.dat",
        MITBIH_DIR / "100.hea",
        MITBIH_DIR / "100.atr",
        MITBIH_DIR / "RECORDS"
    ]

    if all(file.exists() for file in needed_files):
        print("MIT-BIH already exists. Skipping download.")
        return

    print("Downloading MIT-BIH dataset...")
    wfdb.dl_database("mitdb", dl_dir=str(MITBIH_DIR))
    print("MIT-BIH download finished.")