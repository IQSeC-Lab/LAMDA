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


