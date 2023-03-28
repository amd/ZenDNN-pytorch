import csv
from tabulate import tabulate

table_data = []
headers = ["Suite", "TorchDynamo", "TorchScript"]

for suite in ["tb", "hf", "timm"]:
    dynamo_status = {}
    ts_status = {}

    with open(f'../benchmark_logs/{suite}_dynamo.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row['batch_size']) != 0 and int(row['graphs']) != 0 and int(row['graph_calls']) != 0 and int(row['captured_ops']) != 0:
                dynamo_status[row['name']] = "pass"
            else:
                dynamo_status[row['name']] = "fail"
                print(row["name"])

    with open(f'../benchmark_logs/{suite}_ts.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['status'] == '0.0000' or row['status'] == 'fail':
                ts_status[row['name']] = "fail"
            else:
                ts_status[row['name']] = "pass"

    def calculate_passrate(model_dict):
        pass_count = 0
        total_count = len(model_dict)
        for key, value in model_dict.items():
            if value == 'pass':
                pass_count += 1
        pass_rate = (pass_count * 1.0) / total_count
        pass_rate = round(pass_rate * 100, 2)
        return pass_rate

    dynamo_passrate = calculate_passrate(dynamo_status)
    tb_passrate = calculate_passrate(ts_status)
    table_data.append([suite, f"{dynamo_passrate}%", f"{tb_passrate}%"])

print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

# Generate LaTeX code for the table
latex_code = tabulate(table_data, headers=headers, tablefmt="latex_booktabs")
print(latex_code)
