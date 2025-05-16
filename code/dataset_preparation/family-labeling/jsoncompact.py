import os
import sys
import subprocess
import concurrent.futures

FAILED_LOG = "compactfailed.log"

def process_file(file):
    try:
        # Run jq to compact the JSON file
        result = subprocess.run(
            ['jq', '-c', '.', file],
            check=True,  # Raise an exception if jq fails
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Write the compacted JSON back to the file
        with open(file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)

    except subprocess.CalledProcessError as e:
        # If jq fails, log the file path and error message
        with open(FAILED_LOG, 'a', encoding='utf-8') as log:
            log.write(f"Failed: {file} - Error: {e.stderr}\n")

def find_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def main():
    if len(sys.argv) != 2:
        print("Usage: python jsoncompact.py <directory_with_json_files>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    json_files = find_json_files(directory)

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_file, json_files)

if __name__ == '__main__':
    main()
