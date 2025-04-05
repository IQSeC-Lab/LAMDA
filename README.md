# MAD-CD: Malware Android Dataset with Concept Drift
This repository contains the dataset and code for our research on concept drift in Android malware detection. MAD-CD is designed to help researchers analyze the evolving nature of Android malware by capturing temporal variations and distribution shifts over time.
### Steps to replicate the dataset creation process
1. **Ensure that the input CSV file is available**:
   - Download the latest dataset from AndroZoo: [latest_with-added-date.csv.gz](https://androzoo.uni.lu/static/lists/latest_with-added-date.csv.gz).

2. **Run the `AndroZoo-extractor.py`**:
   - This will process the CSV file and categorize the data into three groups based on `vt_detection`:
     - **Benign**: `vt_detection == 0`
     - **Malware**: `vt_detection >= 4`
     - **Unsure**: `1 <= vt_detection <= 3`
   - SHA-256 hashes for each category and year will be written into separate text files.
   - An HTML report summarizing the count of hashes per category and year will be generated.
3. **Run `APKDownloader.sh` to download samples as per need**:
	- Usage: `bash APKDownloader.sh /benign/2018/hashes.txt 2018 ben 6433e6c4d71ad6c89XXXXX`
4.
