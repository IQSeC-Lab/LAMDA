import pandas as pd
import numpy as np

# experiment prefixes
experiments = [
    'class_il_experiment_cumulative',
    'class_il_experiment_expreplay',
    'class_il_experiment_naive',
    'domain_il_experiment_cumulative',
    'domain_il_experiment_expreplay',
    'domain_il_experiment_naive'
]

for exp in experiments:
    try:
        # Load the 3 run files
        files = [f"{exp}.csv", f"{exp}_run2.csv", f"{exp}_run3.csv"]
        dfs = [pd.read_csv(f) for f in files]

        # Merge avg_f1 by experience
        merged = pd.concat([df[['experience', 'avg_f1']] for df in dfs], axis=0)

        # Group by experience and compute mean & SEM
        grouped = merged.groupby('experience')['avg_f1']
        result = pd.DataFrame({
            'mean_f1': grouped.mean(),
            'sem_f1': grouped.sem()
        }).reset_index()

        # Print header and table
        print(f"\n=== {exp} ===")
        print(result.to_string(index=False))
    except FileNotFoundError as e:
        print(f"Missing file for {exp}: {e.filename}")
