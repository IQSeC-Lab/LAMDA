#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <hash_file> <year> <mal|ben)> <androzoo_api_key>"
    echo "Example: $0 /benign/2018/hashes.txt 2018 ben 6433e6c4d71ad6c89XXXXX"

    exit 1
fi

HASH_FILE="$1"
YEAR="$2"
MALWARE="$3"
API_KEY="$4"

# apk storage directory based on  flag
if [[ "$MALWARE" == "mal" ]]; then
    BASE_DIR="./androzoo_malware"
elif [[ "$MALWARE" == "ben" ]]; then
    BASE_DIR="./androzoo_benign"
else
    echo "Invalid malware flag. Use 'mal' or 'ben'."
    exit 1
fi

OUTPUT_DIR="$BASE_DIR/$YEAR"
mkdir -p "$OUTPUT_DIR"

LOG_DIR="./logs/$YEAR"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/downloaded_${MALWARE}.log"
FAILED_LOG="$LOG_DIR/failed_${MALWARE}.log"

MAX_CONCURRENT=40 # max 40 outside europe
CURRENT_JOBS=0

# Ensure log files exist
touch "$LOG_FILE" "$FAILED_LOG"

download_apk() {
    local sha256
    sha256=$(echo "$1" | tr '[:upper:]' '[:lower:]')  # Convert to lowercase

    # Skip if already downloaded
    if grep -q "^$sha256$" "$LOG_FILE"; then
        return
    fi

    # Download 
    curl -G --silent --fail --remote-header-name \
        -d apikey="$API_KEY" \
        -d sha256="$sha256" \
        "https://androzoo.uni.lu/api/download" \
        -o "$OUTPUT_DIR/$sha256.apk"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        # integrity SHA256 hash check
        downloaded_sha256=$(sha256sum "$OUTPUT_DIR/$sha256.apk" | awk '{print $1}')
        
        if [ "$downloaded_sha256" == "$sha256" ]; then
            # Log successful download
            echo "$sha256" >> "$LOG_FILE"
        else
            # If hashes don't match, log as failed
            echo "$sha256" >> "$FAILED_LOG"
            rm -f "$OUTPUT_DIR/$sha256.apk"  # Optionally remove the corrupted file
        fi
    else
        # Log failure if curl fails
        echo "$sha256" >> "$FAILED_LOG"
    fi
}

echo "Processing: $HASH_FILE"
echo "Saving APKs to: $OUTPUT_DIR"
echo "Logs in: $LOG_DIR"

while IFS= read -r sha256; do
    download_apk "$sha256" &
    ((CURRENT_JOBS++))

    if ((CURRENT_JOBS >= MAX_CONCURRENT)); then
        wait -n
        ((CURRENT_JOBS--))
    fi
done < "$HASH_FILE"

wait

echo "Download complete."
