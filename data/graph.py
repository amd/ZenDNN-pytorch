import matplotlib.pyplot as plt
import math

def parse_log(file_name, slug):
    out = []
    with open(file_name, "r") as f:
        for line in f:
            if line.startswith(slug):
                out.append(line)
    def process_item(name, x):
        # print(name)
        if x is not None:
            if 'ERROR' in x or 'Traceback' in x or 'function:' in x:
                return 1.0
            else:
                # print(x)
                as_float = float(x.replace('x', ''))
                if as_float < 1.0:
                    return 1.0
                else:
                    return as_float
        return 1.0

    # print(file_name)
    out = [[item.split()[0], item.split()[1], item.split()[2], process_item(item.split()[0], item.split()[3])] for item in out]
    return out

def describe_delta(data1, data2, name1, name2, run_name):
    model_names = []
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        assert row1[2] == row2[2]  # Make sure model names match between the two datasets
        model_names.append(row1[2])
        if row1[3] is not None:
            performances1.append(row1[3])
        if row2[3] is not None:
            performances2.append(row2[3])
    
    print(f"==== {run_name} ====")
    print(f"Geomean {name1} : {sum(performances1)/len(performances1)}")
    print(f"Geomean {name2} : {sum(performances2)/len(performances2)}")
    
    
def graph_delta(data_names, title):
    data = []
    names = []

    # Plot the performances for the datasets
    fig, ax = plt.subplots(figsize=(24, 10))
    fig.subplots_adjust(bottom=0.45)
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels

    width = 0.25

    last_names = []
    i = 0
    for (d, n) in data_names:
        # Extract the model names and their corresponding performances from the data
        model_names = []
        performances = []
        for row in d:
            model_names.append(row[2])
            performances.append(row[3])

        names.append(n)

        breakpoint()
        ax.bar(model_names, performances, width=width, align='edge', label=n)
        i += 1

    # ax.axhline(y=1, color='r', linestyle='--')
    ax.set_ylim(bottom=1.0)
    ax.set_xticklabels(model_names, rotation=90, ha='right')
    ax.set_ylabel('Speedup')

    # ax.set_title('PT2 Cuda Eval Backend Comparison - HF')
    ax.legend()
    name = f"data/{title}.png"
    print(f"Saving to {name}")
    plt.savefig(name, bbox_inches='tight')

def graph_delta_merged(data1, data2, name1, name2, title):
    # Extract the model names and their corresponding performances from the data
    model_name_to_perf1 = {}
    model_name_to_perf2 = {}
    performances1 = []
    performances2 = []

    for row1, row2 in zip(data1, data2):
        model_name_to_perf1[row1[2]] = row1[3]
        model_name_to_perf2[row2[2]] = row2[3]


    model_names = []
    for name in model_name_to_perf1.keys():
        model_names.append(name)
        performances1.append(model_name_to_perf1[name])
        if name in model_name_to_perf2:
            performances2.append(model_name_to_perf2[name])
        else:
            performances2.append(0)


    for name in model_name_to_perf2.keys():
        if name in model_names:
            continue 
        model_names.append(name)
        performances1.append(0)
        performances2.append(model_name_to_perf2[name])



    # Plot the performances for the two datasets
    fig, ax = plt.subplots(figsize=(24, 12))
    fig.subplots_adjust(bottom=0.45)
    plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
    ax.bar(model_names, performances1, width=-0.4, align='edge', label=name1)
    ax.bar(model_names, performances2, width=0.4, align='edge', label=name2)
    ax.margins(x=0)
    # ax.axhline(y=1, color='r', linestyle='--')
    bottom = 4
    top = math.ceil(max(max(performances2), max(performances1))) * 4
    ticks = []
    labels = []
    for i in range(bottom, top):
        ticks.append(i / 4)
        labels.append(f"{i / 4}x")
    
    ax.set_ylim(bottom=1.0, top=top/4)
    ax.set_xticklabels(model_names, rotation=90, ha='right')
    ax.set_yticks(ticks, labels)
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
        inductor_eval = parse_log(f"data/{bench}_inductor_eval.log", f"cuda eval")
        inductor_train = parse_log(f"data/{bench}_inductor_train.log", f"cuda train")
        breakpoint()
        graph_delta_merged(inductor_eval, inductor_train, "Inference", "Training", title=f"{bench}_inductor_per_kind")


# draw_per_kind_per_bench_graphs()

def draw_all_not_merged():
    benches = ["hf", "timm", "tb"]
    kinds = ["eval", "train"]
    inductor = []
    # nvfuser = []
    for bench in benches:
        for kind in kinds:
            inductor.extend(parse_log(f"data/{bench}_inductor_{kind}.log", f"cuda {kind}"))
            # nvfuser.extend(parse_log(f"data/{bench}_nvfuser_{kind}.log", f"cuda {kind}"))
    # describe_delta(inductor, nvfuser, "Inductor", "NVFuser", f"all_inductor_nvfuser_gpu")
    graph_delta([(inductor, "inductor")], f"all_inductor_gpu")

def draw_all_merged():
    benches = ["hf", "timm", "tb"]
    kinds = ["eval", "train"]
    inductor = []
    # nvfuser = []
    for bench in benches:
        for kind in kinds:
            inductor.extend(parse_log(f"data/{bench}_inductor_{kind}.log", f"cuda {kind}"))
            # nvfuser.extend(parse_log(f"data/{bench}_nvfuser_{kind}.log", f"cuda {kind}"))
    
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
                subseq = grouped_dict[key][i][-1]
                val = curr + subseq
                merged_entry[-1] = val / 2
            merged_list.append(merged_entry)

        return merged_list

    # describe_delta(merge(inductor), merge(nvfuser), "Inductor", "NVFuser", f"merged_inductor_nvfuser_gpu")
    # describe_delta(merge(inductor), "Inductor", "NVFuser", f"all_inductor_gpu")
    graph_delta([(merge(inductor), "Inductor")], f"merged_inductor_gpu")


draw_per_kind_per_bench_graphs()
# draw_all_not_merged()
# draw_all_merged()
