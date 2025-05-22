import pandas as pd
import argparse

def process_csv(input_csv, output_csv):
    
    combined_df = pd.read_csv(input_csv)

    logs = []

    # store the final results
    final_results = []

    # process data by experience
    for experience in combined_df['experience'].unique():

        exp_df = combined_df[combined_df['experience'] == experience]
        #print(experience)
        # group by mode (backward/forward)
        for mode in exp_df['mode'].unique():
            mode_df = exp_df[exp_df['mode'] == mode]
            #print(mode)
            length = len(mode_df)
            
            # Log the experience, mode, and strategy information
            logs.append(f"{mode} - Strategy: {mode_df['strategy'].iloc[0]} Experience: {experience} - length for backward N: {len(exp_df[exp_df['mode'] == 'backward'])} - length for forward N: {len(exp_df[exp_df['mode'] == 'forward'])}")
            
            # Calculate the averages for accuracy, precision, recall, and f1 score
            avg_accuracy = mode_df['accuracy'].mean()
            #print(mode_df['accuracy'])
            avg_precision = mode_df['precision'].mean()
            avg_recall = mode_df['recall'].mean()
            avg_f1 = mode_df['f1'].mean()

            # Store the results in a final list
            final_results.append({
                'experience': experience,
                'mode': mode,
                'avg_accuracy': avg_accuracy,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1': avg_f1,
                'N': length
            })

    final_results_df = pd.DataFrame(final_results)

    final_results_df.to_csv(output_csv, index=False)
    
    # print logs
    for log in logs:
        print(log)
    
    return final_results_df


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Process a CSV file for model results')
    parser.add_argument('input_csv', type=str, help='Path to the CSV file with results')
    parser.add_argument('output_csv', type=str, help='Path to save the averaged results CSV')
    args = parser.parse_args()

    # Call the function to process the CSV
    process_csv(args.input_csv, args.output_csv)
