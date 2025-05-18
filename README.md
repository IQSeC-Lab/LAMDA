# LAMDA: A Longitudinal Android Malware Benchmark for Concept Drift Analysis
This repository contains the dataset and code for our research on concept drift in Android malware detection. **LAMDA** is designed to help researchers analyze the evolving nature of Android malware by capturing temporal variations and distribution shifts over time.
Our dataset is publicly available on Hugging Face:

[**LAMDA Dataset**](https://huggingface.co/datasets/IQSeC-Lab/LAMDA)

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
5. **Now, we are ready with features. Let's extract the family labels with the help of Virustotal and AVClass**.
   - First collect academic API access for the virustotal
   - Use the virustotal-downloader.py to download all reports from virustotal
   - Usage: `python virustotal-downloader.py all_year_malware_hashes.txt`
   - To make these JSON files usable for AVClass, we need to convert them into compact JSONL format.
   - Use the jsoncompact.py to compact all the json file
   - Usage: `python jsoncompact.py <directory_with_json_files>`
   - Install AVClass from [here](https://github.com/malicialab/avclass.git)
   - Now run avclass to get the family labels
   - Usage `avclass -d ./all_year_vt_json/ -hash sha256 -o labels.txt`
   - Now we are ready with features and family labels.

## Section 3: LAMDA Creation
After downloading and decompiling all the APKs, the creation of **LAMDA** is the following:

First, you need to split all the .data files into train and test using the following code. Make sure you have all the .data files into yearly folders as followings and you have provided the correct metadata.csv file which you can find from LAMDA huggingface repository [here](https://huggingface.co/datasets/IQSeC-Lab/LAMDA/tree/main)
   ```
   ├── malware/
   │   ├── 2013/hashes.data ...
   │   ├── 2014/hashes.data ...
   │   └── ...
   ├── benign/
   │   ├── 2013/hashes.data ...
   │   ├── 2014/hashes.data ...
   │   └── ...
   ```

   ```
   python LAMDA_get_train_test.py
   ```
   After the execution is done, you will get the train and test splits saved into npz.

For vectorization and final dataset creation, you need to use the following code. Make sure that you have provided the correct directory ( where the train test split npz files you get from the previous run) to the following code.
   ```
   python vectorization_npz_creation.py
   ```

After vectorization, you will be able to get final LAMDA dataset which is saved into npz files. All the features are saved in YYYY_X_train.npz and metadata containing binary label, family label, number of virus total flagged, year month and hash are saved in YYYY_meta_train.npz (YYYY = four digit of the year when the APK was added to AndroZoo repository) 

Next, you might need to split yearly npz files into monthly npz file, if you want it. For LAMDA experiments and concept drift detection, you will need to split them into yearly form.  



## Section 4: Concept Drift Analysis
For LAMDA experiments, go through the following instructions to recreate the results

1. **First, download LAMDA from LAMDA huggingface repository**
   The following code will help you download LAMDA from huggingface repository, convert the parquet file to npz, split yearly npz into monthly npz for concept drift analysis on LAMDA
   ```
   python download_dataset_and_convert_to_npz.py
   ```

2. **Second, concept drift analysis with supervised learning**
   Use the following code to run the experiments with supervised learning on LAMDA under AnoShift-style splits, make sure you have provided the correct path of downloaded LAMDA

   ```
   python anoshift_experiment_models_separate.py
   ```
   To run all models in one command use the following code script,
   ```
   ./4_1_anoshift_script_LAMDA.sh
   ```

3. **Plots and visualization**
   F1-scores across the years for four models
   ```
   python plot_near_far.py
   ```
   For the histogram, t-sne and Jeffreys divergence heatmap plot which is provided in the paper, use the followings:
   ```
   python malware_family_historgram.py
   python 4_1_visual_analysis_jeffreys_divergence.py
   python 4_1_visual_analysis_t_sne.py
   ```

4. **To compare with API Graph dataset, you need to download API Graph dataset**
   You can download the API Graph dataset from [here](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view)
   Use the code following code for the same set of experiments as previously mentioned
   ```
   python anoshift_experiment_api_graph.py
   ```
   
   We select the top 100 malware families from the LAMDA dataset and compute their stability scores over time. These scores are then compared with the corresponding families found in the APIGraph dataset. The code below generates a combined box plot illustrating the stability scores for both LAMDA and APIGraph. It also includes feature analysis based on Optimal Transport Dataset Distance (OTDD) for both datasets. Further details can be found in Sections 4.3 of the paper.

   ```
   4_3_lambda_stability_otdd_analysis.ipynb
   4_3_apigraph_stability_otdd_analysis.ipynb
   ```

   Furthermore we get 10 common families across all years (2013-2024) and we calculate the feature stability for each individual family from the year 2013 to year 2024. We add the cade evaluation result on test data (2014-2024) in terms of Drift and Non-drift samples. Details are available in the paper Section 4.4. 

   ```
   4_4_stability_analysis_common_families.ipynb
   4_4_cade_evaluation_common_families.ipynb
   ```

