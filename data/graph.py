import matplotlib.pyplot as plt


def parse_log(file_name, slug):
    out = []
    with open(file_name, "r") as f:
        for line in f:
            if line.startswith("slug"):
                out.append(line)
    out = [[item.split()[0], item.split()[1], item.split()[2], item.split()[3]] for item in out]
    return out

def describe_delta(data1, data2, name1, name2):
    for d in [data1, data2]:
        for row in d:
            if row[3] and 'ERROR' in row[3]:
                row[3] = None
    
    model_names = []
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        assert row1[2] == row2[2]  # Make sure model names match between the two datasets
        model_names.append(row1[2])
        if row1[3] is not None:
            performances1.append(float(row1[3].replace('x', '')))
        if row2[3] is not None:
            performances2.append(float(row2[3].replace('x', '')))
    
    print(f"Geomean {name1} : {sum(performances1)/len(performances1)}")
    print(f"Geomean {name2} : {sum(performances2)/len(performances2)}")
    
    
def graph_delta(data1, data2, name1, name2):
    # Filter out the rows with bad data
    for d in [data1, data2]:
        for row in d:
            if row[3] and 'ERROR' in row[3]:
                row[3] = None

    # Extract the model names and their corresponding performances from the data
    model_names = []
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        assert row1[2] == row2[2]  # Make sure model names match between the two datasets
        model_names.append(row1[2])
        performances1.append(float(row1[3].replace('x', '')) if row1[3] is not None else float('nan'))
        performances2.append(float(row2[3].replace('x', '')) if row2[3] is not None else float('nan'))

    # Plot the performances for the two datasets
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(model_names, performances1, width=-0.4, align='edge', label=name1)
    ax.bar(model_names, performances2, width=0.4, align='edge', label=name2)
    ax.axhline(y=1, color='r', linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Speedup')
    # ax.set_title('PT2 Cuda Eval Backend Comparison - HF')
    ax.legend()
    plt.savefig(f'{name1}_{name2}.png')


# Inductor, eval
hf_inductor_eval = "data/hf_inductor_eval.log"
timm_inductor_eval = "data/timm_inductor_eval.log"
tb_inductor_eval = "data/tb_inductor_eval.log"
# NVFuser, eval
hf_nvfuser_eval = "data/hf_nvfuser_eval.log"
timm_nvfuser_eval = "data/timm_nvfuser_eval.log"
tb_nvfuser_eval = "data/tb_nvfuser_eval.log"
# Inductor, train
hf_inductor_train = "data/hf_inductor_train.log"
timm_inductor_train = "data/timm_inductor_train.log"
tb_inductor_train = "data/tb_inductor_train.log"
# NVFuser, train
hf_nvfuser_train = "data/hf_nvfuser_train.log"
timm_nvfuser_train = "data/timm_nvfuser_train.log"
tb_nvfuser_train = "data/tb_nvfuser_train.log"