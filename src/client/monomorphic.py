import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.utils.utils import read_config_file
import statistics
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    
    return parser.parse_args()

args = get_args()
config = read_config_file( args.config_path)

with open(config["TEST_PROGRAMS_LIST"], "r") as f:
    output = np.load(f"prediction.npy")
    cnt = 0
    all_precision = []
    all_recall = []
    all_f1 = []
    for line in f:
        filename = line.strip()
        file_path = os.path.join(config["BENCHMARK_CALLGRAPHS"], filename, config["FULL_FILE"])
        df = pd.read_csv(file_path)
        sa_fanout = defaultdict(lambda: 0)
        new_fanout = defaultdict(lambda: 0)
        da_fanout = defaultdict(lambda: 0)
        call_sites = set()
        for i in tqdm(range(len(df['wiretap']))):
            call_site = (df['method'][i], df['offset'][i], filename)
            call_sites.add(call_site)
            if output[cnt] >= 0.5:
                new_fanout[call_site] += 1

            if df['wiretap'][i] == 1:
                da_fanout[call_site] += 1
            if df[config["SA_LABEL"]][i] == 1:
                sa_fanout[call_site] += 1
                    
            cnt += 1
        
        new_monomorphic_calls = []
        da_monomorphic_calls = []
        sa_monomorphic_calls = []
            
        for cs in call_sites:
            if new_fanout[cs] == 1:
                new_monomorphic_calls.append(1)
            else:
                new_monomorphic_calls.append(0)
                
            if da_fanout[cs] == 1:
                da_monomorphic_calls.append(1)
            else:
                da_monomorphic_calls.append(0)
                
            if sa_fanout[cs] == 1:
                sa_monomorphic_calls.append(1)
            else:
                sa_monomorphic_calls.append(0)
            
        new_monomorphic_calls = np.array(new_monomorphic_calls)
        da_monomorphic_calls = np.array(da_monomorphic_calls)
        doop_monomorphic_calls = np.array(sa_monomorphic_calls)
        temp = precision_score(da_monomorphic_calls, new_monomorphic_calls), recall_score(da_monomorphic_calls, new_monomorphic_calls)
        all_precision.append(temp[0])
        all_recall.append(temp[1])
        all_f1.append(2*temp[0]*temp[1]/(temp[0]+temp[1]))

print("=== Monomorphic Call-site Detection ===")      
print("[EVAL-AVG] Precision {} ({}), Recall {}({}), F1 {}({})".format(round(statistics.mean(all_precision), 2),
                                                                                            round(statistics.stdev(all_precision), 2), 
                                                                                            round(statistics.mean(all_recall), 2),
                                                                                            round(statistics.stdev(all_recall), 2),
                                                                                            round(statistics.mean(all_f1), 2),
                                                                                            round(statistics.stdev(all_f1), 2)))

        
