import hashlib
import os
import zipfile

import requests
from tqdm import tqdm

ML_32M_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ML_32M_MD5 = None  # set to known MD5 if you want checksum verification

REQUIRED_FILES = ["ratings.csv", "tags.csv", "movies.csv", "links.csv"]


def download_movielens_32m(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "ml-32m.zip")

    if verify_files(os.path.join(target_dir, "ml-32m")):
        print("MovieLens 32M already present — skipping download.")
        return

    print(f"Downloading MovieLens 32M from {ML_32M_URL} ...")
    response = requests.get(ML_32M_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="ml-32m.zip"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            bar.update(len(chunk))

    if ML_32M_MD5:
        _verify_md5(zip_path, ML_32M_MD5)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    os.remove(zip_path)
    print(f"Dataset ready at {os.path.join(target_dir, 'ml-32m')}")


def verify_files(data_dir):
    """Return True if all required CSV files are present."""
    if not os.path.isdir(data_dir):
        return False
    for fname in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(data_dir, fname)):
            return False
    return True


def _verify_md5(path, expected_md5):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    actual = md5.hexdigest()
    if actual != expected_md5:
        raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "./data"
    download_movielens_32m(target)
