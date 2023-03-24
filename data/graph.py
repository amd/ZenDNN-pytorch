import matplotlib.pyplot as plt


def parse_log(file_name, slug):
    out = []
    with open(file_name, "r") as f:
        for line in f:
            if line.startswith(slug):
                out.append(line)
    out = [[item.split()[0], item.split()[1], item.split()[2], item.split()[3]] for item in out]
    return out

def describe_delta(data1, data2, name1, name2, run_name):
    for d in [data1, data2]:
        for row in d:
            if row[3] and ('ERROR' in row[3] or 'Traceback' in row[3]):
                row[3] = None
    
    model_names = []
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        assert row1[2] == row2[2]  # Make sure model names match between the two datasets
        model_names.append(row1[2])
        if row1[3] is not None:
            try:
                performances1.append(float(row1[3].replace('x', '')))
            except:
                pass
        if row2[3] is not None:
            try:
                performances2.append(float(row2[3].replace('x', '')))
            except:
                pass
    
    print(f"==== {run_name} ====")
    print(f"Geomean {name1} : {sum(performances1)/len(performances1)}")
    print(f"Geomean {name2} : {sum(performances2)/len(performances2)}")
    
    
def graph_delta(data1, data2, name1, name2, title):
    # Filter out the rows with bad data
    for d in [data1, data2]:
        for row in d:
            if row[3] and ('ERROR' in row[3] or 'Traceback' in row[3]):
                row[3] = None

    # Extract the model names and their corresponding performances from the data
    model_names = []
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        assert row1[2] == row2[2], f"row1: {row1}, row2: {row2}"  # Make sure model names match between the two datasets
        model_names.append(row1[2])
        try:
            performances1.append(float(row1[3].replace('x', '')) if row1[3] is not None else float('nan'))
        except:
            performances1.append(float('nan'))
        try:
            performances2.append(float(row2[3].replace('x', '')) if row2[3] is not None else float('nan'))
        except:
            performances2.append(float('nan'))

    # Plot the performances for the two datasets
    fig, ax = plt.subplots(figsize=(24, 14))
    plt.rc('xtick', labelsize=4)    # fontsize of the tick labels
    ax.bar(model_names, performances1, width=-0.4, align='edge', label=name1)
    ax.bar(model_names, performances2, width=0.4, align='edge', label=name2)
    ax.axhline(y=1, color='r', linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Speedup')
    # ax.set_title('PT2 Cuda Eval Backend Comparison - HF')
    ax.legend()
    name = f"data/{title}.png"
    print(f"Saving to {name}")
    plt.savefig(name, bbox_inches='tight')


benches = ["hf", "timm", "tb"]
kinds = ["eval", "train"]
for bench in benches:
    for kind in kinds:
        try:
            inductor = parse_log(f"data/{bench}_inductor_{kind}.log", f"cuda {kind}")
            nvfuser = parse_log(f"data/{bench}_nvfuser_{kind}.log", f"cuda {kind}")
            describe_delta(inductor, nvfuser, "Inductor", "NVFuser", f"{bench}_inductor_nvfuser_{kind}_gpu")
            graph_delta(inductor, nvfuser, "Inductor", "NVFuser", f"{bench}_inductor_nvfuser_{kind}_gpu")
        except Exception as e:
            print(f"Failed {bench} {kind}")
            raise e

# describe_delta(hf_inductor_eval, hf_nvfuser_eval, "Inductor", "NVFuser", "hf_inductor_nvfuser_inference_gpu")
# # describe_delta(timm_inductor_eval, timm_nvfuser_eval, "Inductor", "NVFuser", "timm_inductor_nvfuser_inference_gpu")
# # describe_delta(tb_inductor_eval, tb_nvfuser_eval, "Inductor", "NVFuser", "tb_inductor_nvfuser_inference_gpu")

# # describe_delta(hf_inductor_train, hf_nvfuser_train, "Inductor", "NVFuser", "hf_inductor_nvfuser_train_gpu")
# # describe_delta(timm_inductor_train, timm_nvfuser_train, "Inductor", "NVFuser", "timm_inductor_nvfuser_train_gpu")
# describe_delta(tb_inductor_train, tb_nvfuser_train, "Inductor", "NVFuser", "tb_inductor_nvfuser_train_gpu")

# # graph_delta(hf_inductor_eval, hf_nvfuser_eval, "Inductor", "NVFuser", "hf_inductor_nvfuser_inference_gpu")
# # graph_delta(timm_inductor_eval, timm_nvfuser_eval, "Inductor", "NVFuser", "timm_inductor_nvfuser_inference_gpu")
# # graph_delta(tb_inductor_eval, tb_nvfuser_eval, "Inductor", "NVFuser", "tb_inductor_nvfuser_inference_gpu")

# # graph_delta(hf_inductor_train, hf_nvfuser_train, "Inductor", "NVFuser", "hf_inductor_nvfuser_train_gpu")
# # graph_delta(timm_inductor_train, timm_nvfuser_train, "Inductor", "NVFuser", "timm_inductor_nvfuser_train_gpu")
# graph_delta(tb_inductor_train, tb_nvfuser_train, "Inductor", "NVFuser", "tb_inductor_nvfuser_train_gpu")
