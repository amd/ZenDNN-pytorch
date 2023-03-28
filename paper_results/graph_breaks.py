# Script generated through chatGPT

import pandas as pd
from tabulate import tabulate

# Define a function to process each CSV file and return the results as a pandas DataFrame
def process_csv_file(suite_name, file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Get the number of graphs for each model
    num_graphs = df['graphs'].tolist()

    # Count the number of models falling into each category
    num_graphs_1 = num_graphs.count(1)
    num_graphs_10 = len([n for n in num_graphs if n <= 10])
    num_graphs_100 = len([n for n in num_graphs if n <= 100])

    # Calculate the percentage of models in each category
    pct_graphs_1 = num_graphs_1 / len(num_graphs) * 100
    pct_graphs_10 = num_graphs_10 / len(num_graphs) * 100
    pct_graphs_100 = num_graphs_100 / len(num_graphs) * 100


    ngraphs = len(num_graphs)
    # Create a table with the results
    table = pd.DataFrame({
        'Suite': [suite_name],
        'graphs $\eq$ 1': [f'{num_graphs_1}/{ngraphs} ({pct_graphs_1:.2f}\%)'],
        'graphs $\leq$ 10': [f'{num_graphs_10}/{ngraphs} ({pct_graphs_10:.2f}\%)'],
        'graphs $\leq$ 100': [f'{num_graphs_100}/{ngraphs} ({pct_graphs_100:.2f}\%)']
    })

    # Return the table with the results
    return table

# Process each CSV file and combine the results into one table
table = pd.concat([
    process_csv_file('tb', '../benchmark_logs/tb_dynamo.csv'),
    process_csv_file('hf', '../benchmark_logs/hf_dynamo.csv'),
    process_csv_file('timm', '../benchmark_logs/timm_dynamo.csv')
], ignore_index=True)

print(f'\nResults:')
print(tabulate(table, headers='keys', tablefmt='orgtbl', showindex=False))
# Generate the LaTeX code to print the table
latex = table.to_latex(index=False, escape=False)

# Print the LaTeX code to the console
print(latex)
