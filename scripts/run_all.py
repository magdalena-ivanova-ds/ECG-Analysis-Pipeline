from config import (
    RAW_DIR,
    PROCESSED_DIR,
    MITBIH_DIR,
    PTBXL_DIR,
    MODEL1_DIR,
    MODEL2_DIR
)
from utils_ecg import make_dirs
from download_mitbih import download_mitbih
from download_ptbxl import download_ptbxl
from preprocess_mitbih import preprocess_mitbih
from preprocess_ptbxl import preprocess_ptbxl
from create_splits import main as create_splits_main


def main():
    # Create folders
    make_dirs([
        RAW_DIR,
        PROCESSED_DIR,
        MITBIH_DIR,
        PTBXL_DIR,
        MODEL1_DIR,
        MODEL2_DIR
    ])

    # Step 1: download MIT-BIH automatically
    #print("Checking MIT-BIH...")
    #download_mitbih()

    # Step 2: download PTB-XL automatically
    #print("Checking PTB-XL...")
    #download_ptbxl()

    # Step 3: preprocess MIT-BIH for Model 1
    #preprocess_mitbih()

    # Step 4: preprocess PTB-XL for Model 2
    #preprocess_ptbxl()

    # Step 5: create train/val/test splits
    create_splits_main()

    print("All preprocessing is finished.")


if __name__ == "__main__":
    main()