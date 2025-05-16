# Section 4: Concept Drift Analysis
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


