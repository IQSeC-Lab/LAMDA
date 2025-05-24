# Supplementary Sections
## A. Dataset Statistics
Year-wise malware/benign counts and family distributions.

```
python malware_family_histogram.py
```

## C. Model Architectures and Detail Results
Architectures and extended evaluation on LAMDA variants.

```
python anoshift_experiment_models_separate.py
```
or,
```
./anoshift_script.sh
```
Change to appropriate directory (must be npz file)

## E. Effect of Label Noise in Training Data
Impact of different VirusTotal thresholds on labeling.

```
python label_noise.py
```
To run this, you need to download metadata.csv from huggingface dataset repository.

## F. Label Drift Across Years Based on VirusTotal Label Changes
Year-wise analysis of evolving VirusTotal labels. We have already shared the current AV detection result from Virustotal vs The Androzoo Repository metadata in vt_detections.csv along with the sample added year.
```
python analyze_vt_stats.py.py
```
This script will generate the table that has been provided in the paper. In order to generate a plot based on the data from the previous script run the below script.
```
python plot-detection_drift_by_year.py
```
## I. SHAP-Based Explanation Drift
Temporal trends in top 1000 feature attributions.
```
concept_drift_analysis/4_5_shap_explanation_monthly_lamda.py
```
run this file to generate top 100 and top 1000 SHAP indices from lamda dataset and store in file top_shap_indices_100_lamda.txt and top_shap_indices_1000_lamda.txt
```
concept_drift_analysis/4_5_shap_explanation_monthly_apigraph.py
```
run this file to generate top 100 and top 1000 SHAP indices from apigraph dataset and store in file top_shap_indices_100_apigraph.txt and top_shap_indices_1000_apigraph.txt
```
concept_drift_analysis/4_5_shap_explanation_graphs.py 
```
run this file to generate graphs shap_explanation_drift_monthly_top_100_features_apigraph.png, shap_explanation_drift_monthly_top_1000_features_apigraph.png, shap_explanation_drift_monthly_top_100_features_lamda.png and shap_explanation_drift_monthly_top_1000_features_lamda.png

## J. Continual Learning on LAMDA
Class- and domain-incremental learning benchmarks.
