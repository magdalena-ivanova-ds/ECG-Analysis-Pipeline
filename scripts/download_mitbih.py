import wfdb
from config import mitbih_dir
from utils_ecg import make_dirs


def download_mitbih():
    make_dirs([mitbih_dir])

    # check for a few key files before downloading again
    required_files = [
        mitbih_dir / "100.dat",
        mitbih_dir / "100.hea",
        mitbih_dir / "100.atr",
        mitbih_dir / "RECORDS",
    ]

    if all(path.exists() for path in required_files):
        print("Mit-bih data is already available, skipping download.")
        return

    print("Downloading mit-bih data.")
    wfdb.dl_database("mitdb", dl_dir=str(mitbih_dir))
    print("Mit-bih download completed.")