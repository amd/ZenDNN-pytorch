import csv
from scipy.stats import gmean

def calculate_geomean_speedup(csv_path):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        speedup_list = []
        for row in reader:
            speedup_list.append(float(row['speedup']))
        geomean_speedup = round(gmean(speedup_list), 2)
    return geomean_speedup

tb_speedup = calculate_geomean_speedup("../benchmark_logs/tb_overhead.csv")
hf_speedup = calculate_geomean_speedup("../benchmark_logs/hf_overhead.csv")
timm_speedup = calculate_geomean_speedup("../benchmark_logs/timm_overhead.csv")

table = [
    ["Framework", "GeoMean Speedup"],
    ["Torchbench", f"{tb_speedup:.2f}x"],
    ["HuggingFace", f"{hf_speedup:.2f}x"],
    ["Timm", f"{timm_speedup:.2f}x"]
]

# Print table
max_len = max([len(str(row)) for row in table])
print(f"+{'-'*(max_len+2)}+\n"
      f"| {'':<{max_len}} |\n"
      f"| {'Framework':<{max_len}} | {'Geometric Mean Speedup':<{max_len}} |\n"
      f"| {'':<{max_len}} |\n"
      f"| {table[1][0]:<{max_len}} | {table[1][1]:<{max_len}} |\n"
      f"| {table[2][0]:<{max_len}} | {table[2][1]:<{max_len}} |\n"
      f"| {table[3][0]:<{max_len}} | {table[3][1]:<{max_len}} |\n"
      f"+{'-'*(max_len+2)}+\n")

# Generate LaTeX code for the table
latex_table = "\\begin{table}[h]\n"
latex_table += f"\\centering\n\\caption{{Geometric mean speedup for each framework.}}\n"
latex_table += "\\begin{tabular}{|c|c|}\n"
latex_table += "\\hline\n"
latex_table += f"\\textbf{{Framework}} & \\textbf{{GeoMean Speedup}} \\\\ \n"
latex_table += "\\hline\n"
for row in table[1:]:
    latex_table += f"{row[0]} & {row[1]} \\\\ \n"
latex_table += "\\hline\n"
latex_table += "\\end{tabular}\n"
latex_table += "\\label{tab:speedup}\n"
latex_table += "\\end{table}\n"

print("LaTeX code:\n")
print(latex_table)

