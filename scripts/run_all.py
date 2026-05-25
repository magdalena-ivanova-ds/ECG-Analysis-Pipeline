from config import raw_dir, processed_dir, mitbih_dir, ptbxl_dir, model1_dir, model2_dir
from utils_ecg import make_dirs
from download_mitbih import download_mitbih
from download_ptbxl import download_ptbxl
from preprocess_mitbih import preprocess_mitbih
from preprocess_ptbxl import preprocess_ptbxl
from create_splits import main as create_splits_main


# Change these to True only if you want to run that step again
run_downloads = False
run_preprocessing = False
run_splits = False


def main():
    make_dirs([
        raw_dir,
        processed_dir,
        mitbih_dir,
        ptbxl_dir,
        model1_dir,
        model2_dir,
    ])

    if run_downloads:
        print("Checking MIT-BIH data")
        download_mitbih()

        print("Checking PTB-XL data")
        download_ptbxl()

    if run_preprocessing:
        print("Preprocessing MIT-BIH data")
        preprocess_mitbih()

        print("Preprocessing PTB-XL data")
        preprocess_ptbxl()

    if run_splits:
        print("Creating split files")
        create_splits_main()

    print("Selected pipeline steps completed.")


if __name__ == "__main__":
    main()