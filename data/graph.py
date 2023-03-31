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
    
    
def graph_delta(data_names, title):
    data = []
    names = []

    last_names = []
    for (d, n) in data_names:
        # Filter out the rows with bad data
        for row in d:
            if row[3] and ('ERROR' in row[3] or 'Traceback' in row[3]):
                row[3] = None

        # Extract the model names and their corresponding performances from the data
        model_names = []
        performances = []
        for row in d:
            model_names.append(row[2])
            try:
                performances.append(float(row[3].replace('x', '')) if row[3] is not None else float('nan'))
            except:
                performances.append(float('nan'))

        print(set(model_names) - set(last_names))
        print(set(last_names) - set(model_names))
        breakpoint()
        last_names = model_names
        data.append(performances)
        names.append(n)

    
    assert len(data) == len(names), "The number of data sets must match the number of names"

    # Plot the performances for the datasets
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.subplots_adjust(bottom=0.45)
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels

    width = -0.4
    for i, d in enumerate(data):
        ax.bar(model_names, d, width=width, align='edge', label=names[i])
        width += 0.4

    ax.axhline(y=1, color='r', linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(model_names, rotation=90, ha='right')
    ax.set_ylabel('Speedup')
    # ax.set_title('PT2 Cuda Eval Backend Comparison - HF')
    ax.legend()
    name = f"data/{title}.png"
    print(f"Saving to {name}")
    plt.savefig(name, bbox_inches='tight')

def graph_delta_merged(data1, data2, name1, name2, title):
    # Extract the model names and their corresponding performances from the data
    model_names = []
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        assert row1[2] == row2[2], f"row1: {row1}, row2: {row2}"  # Make sure model names match between the two datasets
        
        if isinstance(row1[3], str) and isinstance(row2[3], str):
            continue
        if row1[3] == 0 and row2[3] == 0:
            continue

        model_names.append(row1[2])
        if isinstance(row1[3], str):
            performances1.append(float('nan'))
        else:
            performances1.append(row1[3])

        if isinstance(row2[3], str):
            performances2.append(float('nan'))
        else:
            performances2.append(row2[3])

    # Plot the performances for the two datasets
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.subplots_adjust(bottom=0.45)
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    ax.bar(model_names, performances1, width=-0.4, align='edge', label=name1)
    ax.bar(model_names, performances2, width=0.4, align='edge', label=name2)
    ax.axhline(y=1, color='r', linestyle='--')
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(model_names, rotation=90, ha='right')
    ax.set_ylabel('Speedup')
    # ax.set_title('PT2 Cuda Eval Backend Comparison - HF')
    ax.legend()
    name = f"data/{title}.png"
    print(f"Saving to {name}")
    plt.savefig(name, bbox_inches='tight')


def draw_per_kind_per_bench_graphs():
    benches = ["hf", "timm", "tb"]
    kinds = ["eval", "train"]
    inductor_eval = []
    nvfuser_eval = []
    inductor_train = []
    nvfuser_train = []
    for bench in benches:
        inductor_eval.extend(parse_log(f"data/{bench}_inductor_eval.log", f"cuda eval"))
        nvfuser_eval.extend(parse_log(f"data/{bench}_nvfuser_eval.log", f"cuda eval"))
        inductor_train.extend(parse_log(f"data/{bench}_inductor_train.log", f"cuda train"))
        nvfuser_train.extend(parse_log(f"data/{bench}_nvfuser_train.log", f"cuda train"))

    graph_delta([(inductor_eval, "Inductor - Inference"), (inductor_train, "Inductor - Train"), (nvfuser_eval, "NVFuser - Inference"), (nvfuser_train, "NVFuser - Train")], title=f"all_inductor_nvfuser_gpu")


# draw_per_kind_per_bench_graphs()

def draw_all_not_merged():
    benches = ["hf", "timm", "tb"]
    kinds = ["eval", "train"]
    inductor = []
    nvfuser = []
    for bench in benches:
        for kind in kinds:
            inductor.extend(parse_log(f"data/{bench}_inductor_{kind}.log", f"cuda {kind}"))
            nvfuser.extend(parse_log(f"data/{bench}_nvfuser_{kind}.log", f"cuda {kind}"))
    describe_delta(inductor, nvfuser, "Inductor", "NVFuser", f"all_inductor_nvfuser_gpu")
    graph_delta([(inductor, "inductor"), (nvfuser, "NVFuser")], f"all_inductor_nvfuser_gpu")

def draw_all_merged():
    benches = ["hf", "timm", "tb"]
    kinds = ["eval", "train"]
    inductor = []
    nvfuser = []
    for bench in benches:
        for kind in kinds:
            inductor.extend(parse_log(f"data/{bench}_inductor_{kind}.log", f"cuda {kind}"))
            nvfuser.extend(parse_log(f"data/{bench}_nvfuser_{kind}.log", f"cuda {kind}"))
    
    def merge(records):
        grouped_dict = {}

        for entry in records:
            if entry[2] not in grouped_dict:
                grouped_dict[entry[2]] = []
            grouped_dict[entry[2]].append(entry)

        merged_list = []

        for key in grouped_dict:
            merged_entry = []
            for i in range(len(grouped_dict[key][0])):
                merged_entry.append(grouped_dict[key][0][i])
            for i in range(1, len(grouped_dict[key])):
                curr = merged_entry[-1]
                first_fail = False
                if isinstance(curr, str):
                    try:
                        val = float(curr.replace("x", ''))
                    except:
                        first_fail = True
                        val = 0
                else:
                    val = curr

                subseq = grouped_dict[key][i][-1]
                if isinstance(curr, str):
                    try:
                        val = val + float(subseq.replace("x", ''))
                    except:
                        first_fail = True
                        val = 0
                else:
                    val = val + subseq
                
                if first_fail:
                    merged_entry[-1] = val 
                else:
                    merged_entry[-1] = val / 2
            merged_list.append(merged_entry)

        return merged_list

    # describe_delta(merge(inductor), merge(nvfuser), "Inductor", "NVFuser", f"merged_inductor_nvfuser_gpu")
    graph_delta_merged(merge(inductor), merge(nvfuser), "Inductor", "NVFuser", f"merged_inductor_nvfuser_gpu")


draw_per_kind_per_bench_graphs()
# draw_all_not_merged()
# draw_all_merged()
