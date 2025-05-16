import requests
import time
import sys
import json
import os
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Constants
MAX_RETRIES = 3
MAX_THREADS = 50
RETRY_DELAY = 2
OUTPUT_DIR = "all_year_vt_json"
LOG_FILE = "./downloaded_hashes.log" # to make surer we dont download same sample twic

# API keys
API_KEYS = [
    "8b2c70XXXXXX5bf183b280f",
    "e3d793XXXXX9305485698d8",
    "6882c1XXXXXX61cc0bc7207",
    "71d737XXXXXXXXX85982397",
    "2fe58b26XXXX75e0796e6c7",
]

# Globals
lock = Lock()
downloaded_hashes = set()
key_cooldown = {key: 0 for key in API_KEYS}

def load_downloaded_hashes(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            return set(line.strip() for line in f.readlines())
    return set()

def append_to_log(log_file, sha256_hash):
    with lock:
        with open(log_file, 'a') as f:
            f.write(f"{sha256_hash}\n")
        downloaded_hashes.add(sha256_hash)

def get_available_key():
    current_time = time.time()
    available_keys = [key for key, cooldown in key_cooldown.items() if cooldown <= current_time]
    if available_keys:
        return random.choice(available_keys)
    return None

def get_virustotal_info(sha256_hash):
    for attempt in range(MAX_RETRIES):
        key = get_available_key()
        if not key:
            time.sleep(RETRY_DELAY)
            continue

        url = f'https://www.virustotal.com/api/v3/files/{sha256_hash}'
        try:
            response = requests.get(url, headers={'x-apikey': key}, timeout=10)

            if response.status_code == 429:
                logging.warning(f"Key {key[:8]}... rate limited, switching keys")
                with lock:
                    key_cooldown[key] = time.time() + 60  # 1 minute cooldown
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed for {sha256_hash}: {str(e)}")
            time.sleep(RETRY_DELAY)

    logging.error(f"Failed all retries for {sha256_hash}")
    return None

def process_hash(hash_info):
    sha256_hash = hash_info

    if sha256_hash in downloaded_hashes:
        logging.info(f"Skipping already downloaded hash: {sha256_hash}")
        return

    response = get_virustotal_info(sha256_hash)
    if response:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"{sha256_hash.lower()}.json")
        with open(output_file, 'w') as f:
            json.dump(response, f)
        append_to_log(LOG_FILE, sha256_hash)
        logging.info(f"Successfully processed {sha256_hash}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python virustotal-downloader.py <malware_hashes_file>")
        sys.exit(1)

    hashes_file = sys.argv[1]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    global downloaded_hashes
    downloaded_hashes = load_downloaded_hashes(LOG_FILE)

    with open(hashes_file, 'r') as f:
        hashes = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        executor.map(process_hash, hashes)

if __name__ == '__main__':
    main()
