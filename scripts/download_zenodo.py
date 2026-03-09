"""
Download models and data from Zenodo archive.
Run once before starting the application.

Usage:
    python download_zenodo.py [--all] [--light]

    --light   Skip BERT models (only TF-IDF + corpus, ~88 MB)
    --all     Download everything including BERT models (~1 GB)
    Default:  Downloads everything (same as --all)
"""

import os
import sys
import hashlib
import zipfile
import urllib.request
from pathlib import Path

ZENODO_RECORD = "18256018"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# File manifest: (filename, md5, size_desc, required_for_light)
FILES = {
    "tfidf_baseline": {
        "url": f"{BASE_URL}/su_adm_tfidf_baseline.zip?download=1",
        "filename": "su_adm_tfidf_baseline.zip",
        "md5": "9ce6ff06ce41a4171f4e2844391454a9",  # from Zenodo — double-check
        "size": "10.9 MB",
        "light": True,
        "extract_to": "../backend/models/tfidf_baseline",
    },
    "preprocessed_data": {
        "url": f"{BASE_URL}/su_adm_preprocessed_data.zip?download=1",
        "filename": "su_adm_preprocessed_data.zip",
        "md5": "9ce6ff06ce41a4171f4e2844391454a9",
        "size": "2.6 MB",
        "light": True,
        "extract_to": "../backend/data/preprocessed",
    },
    "raw_corpus": {
        "url": f"{BASE_URL}/Kursplanekorpus-2023-original-ej-bearb.csv?download=1",
        "filename": "Kursplanekorpus-2023-original-ej-bearb.csv",
        "md5": "aa0a686f7816dd3f7fb7ce386f4cfda0",
        "size": "75.5 MB",
        "light": True,
        "extract_to": None,  # plain file, copy to data dir
        "copy_to": "../backend/data/corpus_raw.csv",
    },
    "bert_binary": {
        "url": f"{BASE_URL}/su_admin_bert_binary.zip?download=1",
        "filename": "su_admin_bert_binary.zip",
        "md5": "975642714101deaa3ae9959ab89da4ba",
        "size": "463.5 MB",
        "light": False,
        "extract_to": "../backend/models/bert_binary",
    },
    "bert_distributional": {
        "url": f"{BASE_URL}/bert_distributional_model.zip?download=1",
        "filename": "bert_distributional_model.zip",
        "md5": "23c23f0e6c878b9a127fb8c4542fe44b",
        "size": "463.2 MB",
        "light": False,
        "extract_to": "../backend/models/bert_distributional",
    },
}

SCRIPT_DIR = Path(__file__).parent
DOWNLOAD_DIR = SCRIPT_DIR / "downloads"


def download_file(url, dest, desc=""):
    """Download a file with progress indicator."""
    print(f"  Downloading {desc}...")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  [{pct:3d}%] {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print()


def extract_zip(zip_path, extract_to):
    """Extract a zip file to target directory."""
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def main():
    light_mode = "--light" in sys.argv
    mode_label = "light (TF-IDF + corpus only)" if light_mode else "full (all models)"
    print(f"\n{'='*60}")
    print(f"SU Admin Demo — Zenodo Downloader")
    print(f"Mode: {mode_label}")
    print(f"{'='*60}\n")

    DOWNLOAD_DIR.mkdir(exist_ok=True)

    for key, info in FILES.items():
        if light_mode and not info["light"]:
            print(f"⏭  Skipping {key} (BERT model, use --all to include)")
            continue

        dest = DOWNLOAD_DIR / info["filename"]

        # Skip if already downloaded
        if dest.exists():
            print(f"✓  {key} already downloaded ({info['size']})")
        else:
            download_file(info["url"], dest, desc=f"{key} ({info['size']})")

        # Extract or copy
        if info.get("extract_to"):
            target = SCRIPT_DIR / info["extract_to"]
            if not target.exists() or not any(target.iterdir()):
                extract_zip(dest, target)
            else:
                print(f"  Already extracted to {target}")
        elif info.get("copy_to"):
            target = SCRIPT_DIR / Path(info["copy_to"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                import shutil
                shutil.copy2(dest, target)
                print(f"  Copied to {target}")

    print(f"\n{'='*60}")
    print("Download complete!")
    if light_mode:
        print("\nNote: BERT models not downloaded. Run with --all to include them.")
        print("Without BERT models, you can still run TF-IDF predictions")
        print("but the pre-compute step will be skipped.")
    else:
        print("\nNext step: python precompute.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
