import cudf
import pandas as pd
import os
import subprocess

# Set the chunk size (number of rows per chunk)
chunk_size = 100000  

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_year_slice(df_year, category, category_name):

    create_dir(f"{category_name}/{category}")
    
    with open(f"{category_name}/{category}/hashes.txt", "a") as category_file:
        hashes = df_year["sha256"].to_arrow().to_pylist()  #  cudf list conversion
        for hash_value in hashes:
            category_file.write(f"{hash_value}\n")
    
    # csv_path = f"{category_name}/{category}/data.csv"
    
    # # wrte header if file doesn't exist, or append without header
    # if not os.path.exists(csv_path):
    #     df_year.to_csv(csv_path, index=False)
    # else:
    #     df_year.to_csv(csv_path, index=False, header=False)

def process_chunk(df_chunk):
    # Ensure the 'added' column is a string type 
    df_chunk["added"] = df_chunk["added"].astype(str)
    
    # Extract year fromfirst 4 char
    df_chunk["year"] = df_chunk["added"].str.slice(0, 4)
    
    # Ensure that extracted years are valid (numeric or valid format)
    df_chunk = df_chunk[df_chunk["year"].str.isnumeric()]

    # Convert the 'year' column to a cuDF Series for iteration
    unique_years = df_chunk["year"].unique().to_pandas()
    # Process each year individually
    for year in unique_years:
        df_year = df_chunk[df_chunk["year"] == year]

        # Process malware (vt_detection >= 4)
        malware_year = df_year[df_year["vt_detection"] >= 4]
        if len(malware_year) > 0:
            print(f"Processing malware for year {year} (rows: {len(malware_year)})")
            process_year_slice(malware_year, year, "malware")

        # Process benign (vt_detection == 0)
        benign_year = df_year[df_year["vt_detection"] == 0]
        if len(benign_year) > 0:
            print(f"Processing benign for year {year} (rows: {len(benign_year)})")
            process_year_slice(benign_year, year, "benign")

        # Process unsure (1 <= vt_detection <= 3)
        unsure_year = df_year[(df_year["vt_detection"] >= 1) & (df_year["vt_detection"] <= 3)]
        if len(unsure_year) > 0:
            print(f"Processing unsure for year {year} (rows: {len(unsure_year)})")
            process_year_slice(unsure_year, year, "unsure")

# Function to generate HTML table report using `wc -l`
def generate_html_report():
    html = "<html><head><title>Yearly Report</title></head><body>"
    html += "<h1>Malware, Benign, and Unsure Count per Year</h1>"
    html += "<table border='1'><tr><th>Year</th><th>Malware</th><th>Benign</th><th>Unsure</th></tr>"
    
    # List categories and years to count the lines in the respective text files
    categories = ["malware", "benign", "unsure"]
    years = range(2013, 2026)  # Adjust the years range based on your data
    
    for year in years:
        html += f"<tr><td>{year}</td>"
        
        for category in categories:
            # Use wc -l to count lines in the respective hash file
            try:
                command = f"wc -l ./{category}/{year}/hashes.txt"
                result = subprocess.check_output(command, shell=True)
                count = int(result.split()[0])
            except subprocess.CalledProcessError:
                count = 0  # In case the file does not exist or an error occurs

            html += f"<td>{count}</td>"
        
        html += "</tr>"
    
    html += "</table></body></html>"
    return html

# Input the androozoo latest with added file
csv_file = "latest_with-added-date.csv"

# Open the CSV and process in chunks
for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
    print(f"Processing chunk {i + 1}...")
    
    # Convert chunk to cuDF DataFrame (GPU accelerated)
    df_chunk = cudf.from_pandas(chunk)
    
    # Process this chunk
    process_chunk(df_chunk)

# Generate and save the HTML report
html_report = generate_html_report()
with open("yearly_report.html", "w") as f:
    f.write(html_report)

print("Processing complete! Files were written to disk, and the HTML report has been generated.")
