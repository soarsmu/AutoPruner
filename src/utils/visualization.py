from dgl.data.utils import makedirs, save_info, load_info
from sklearn.manifold import TSNE
import numpy as np
import argparse
import os
from src.utils.utils import read_config_file
from matplotlib import pyplot as plt
import pandas as pd
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def visualize(code_feats, labels, path):
    tsne = TSNE(n_components=2).fit_transform(code_feats)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors_per_class = {0: "red", 1: "green"}
    for label in colors_per_class:
        indices = [i for i, l in enumerate(labels) if l == label]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        ax.scatter(current_tx, current_ty, c=colors_per_class[label], label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig(path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/doop.config") 
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    info_dict = load_info(os.path.join(config["FINETUNE_EMBEDDING"], "ft_test.pkl"))
    code_feats = np.array(info_dict['code'])
    labels =  np.array(info_dict['target'])
    start_idx = 0
    cnt = 0
    with open(config['TEST_PROGRAMS_LIST'], "r") as f:
        for line in f:
            if True:
                filename = line.strip()
                saved_path = os.path.join(config['VISUALIZATION_OUTPUT'], filename)
                file_path = os.path.join(config["BENCHMARK_CALLGRAPHS"], filename, config["FULL_FILE"])
                df = pd.read_csv(file_path)
                total_sample = len(df['wiretap'])                    
                
                feat = code_feats[start_idx: start_idx + total_sample, :]
                label = labels[start_idx: start_idx + total_sample]
                print(np.sum(df['wiretap']))
                print(np.sum(label))
                print("=====")
                cnt += np.sum(df['wiretap'])
                visualize(feat, label, saved_path)
                # cnt += 1
                start_idx += total_sample
    print(cnt)


if __name__ == '__main__':
    main()


  