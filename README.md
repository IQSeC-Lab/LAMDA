# LAMDA: A Longitudinal Android Malware Benchmark for Concept Drift Analysis
This repository contains the dataset and code for our research on concept drift in Android malware detection. **LAMDA** is designed to help researchers analyze the evolving nature of Android malware by capturing temporal variations and distribution shifts over time.
### Steps to replicate the dataset creation process
1. **Ensure that the input CSV file is available**:
   - Download the latest dataset from AndroZoo: [latest_with-added-date.csv.gz](https://androzoo.uni.lu/static/lists/latest_with-added-date.csv.gz).

2. **Run the `AndroZoo-extractor.py`**:
   - This will process the CSV file and categorize the data into three groups based on `vt_detection`:
     - **Benign**: `vt_detection == 0`
     - **Malware**: `vt_detection >= 4`
     - **Unsure**: `1 <= vt_detection <= 3`
   - SHA-256 hashes for each category and year will be written into separate text files.
   ```
   ├── malware/
   │   ├── 2013/hashes.txt
   │   ├── 2014/hashes.txt
   │   └── ...
   ├── benign/
   │   ├── 2013/hashes.txt
   │   ├── 2014/hashes.txt
   │   └── ...
   ```
   - An HTML report summarizing the count of hashes per category and year will be generated.
3. **Run `APKDownloader.sh` to download samples as per need**:
	- Usage: `bash APKDownloader.sh /benign/2018/hashes.txt 2018 ben YOUR_API_KEY`
   - Another Usage: `bash APKDownloader.sh /malware/2018/hashes.txt 2018 mal YOUR_API_KEY`
4. **After downloading and organizing the APKs into year-wise folders, run the feature extractor to process the APKs.**
   - Usage: 
   ```bash
   python ./dataset_preparation/drebin-feature-extractor/extractor.py \
      --input_dir ./androzoo_malware/2013/ \
      --result_dir ./output/malware/2013/
   ```
   - After completing this process, you will have all the .data files containing Drebin-style features ready.
5. 