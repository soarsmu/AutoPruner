import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from transformers import AutoTokenizer
from dgl.data.utils import save_info, load_info
from tqdm import tqdm
from src.utils.utils import read_config_file
from src.utils.converter import convert
import numpy as np

header_names = [
"-direct#depth_from_main",
"-direct#src_node_in_deg",
"-direct#dest_node_out_deg",
"-direct#dest_node_in_deg",
"-direct#src_node_out_deg",
"-direct#repeated_edges",
"-direct#fanout",
"-direct#graph_node_count", 
"-direct#graph_edge_count",
"-direct#graph_avg_deg",
"-direct#graph_avg_edge_fanout",
"-trans#depth_from_main",
"-trans#src_node_in_deg",
"-trans#dest_node_out_deg",
"-trans#dest_node_in_deg",
"-trans#src_node_out_deg",
"-trans#repeated_edges",
"-trans#fanout",
"-trans#graph_node_count",
"-trans#graph_edge_count",
"-trans#graph_avg_deg",
"-trans#graph_avg_edge_fanout"
]

def compute_header(header_names, header):
    return [header + header_names[i] for i in range(len(header_names))]

class FinetunedDataset(Dataset):
    def __init__(self, config, mode):
        self.mode = mode
        self.config = config
        self.raw_data_path = self.config["BENCHMARK_CALLGRAPHS"]
        self.processed_path = self.config["PROCESSED_DATA"]
        self.save_dir = self.config["CACHE_DIR"]
        self.save_path = os.path.join(self.save_dir, f"ft_{self.mode}.pkl")
        self.cg_file = self.config["FULL_FILE"]
        self.emd_dir = os.path.join(self.save_dir, f"{self.mode}_finetuned")

        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            return NotImplemented
        
        self.header_names = compute_header(header_names, self.config["HEADERS"])
        
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()
    def __len__(self):
        return len(self.code_feats)

    def __getitem__(self, index):
        struct_feats = np.where(self.struct_feats[index] == 1000000000, 100000, self.struct_feats[index])
        return {
            'code': torch.tensor(self.code_feats[index], dtype=torch.float),
            'struct': torch.tensor(struct_feats, dtype=torch.float),
            # 'struct': torch.tensor(self.struct_feats[index], dtype=torch.float),
            'label': torch.tensor(self.labels[index], dtype=torch.long),
            'static': torch.tensor(self.static_ids[index], dtype=torch.float),
            'program_ids': self.program_ids[index]
            }

    def process(self):
        self.code_feats = []
        self.struct_feats = []
        self.labels = []
        self.static_ids = []
        self.program_ids = []
        idx = 0
        program_idx = 0
        with open(self.program_lists, "r") as f:
            for line in f:
                filename = line.strip()
                file_path = os.path.join(self.raw_data_path, filename, self.cg_file)
                df = pd.read_csv(file_path)
                features = df[self.header_names].to_numpy()
                for i in tqdm(range(len(df['wiretap']))):
                    lb, sanity_check = df['wiretap'][i], df[self.config["SA_LABEL"]][i]
                    if self.mode != "train" or sanity_check == 1:
                        batch_idx, in_batch_idx = idx//6, idx%6
                        emb_path = os.path.join(self.emd_dir, f"{batch_idx}.npy")
                        self.code_feats.append(np.load(emb_path)[in_batch_idx])
                        self.struct_feats.append(features[i])
                        self.labels.append(lb)
                        self.static_ids.append(sanity_check)
                        self.program_ids.append(program_idx)
                        idx += 1
                program_idx += 1

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_info(self.save_path, {'code': self.code_feats,
                                    'struct': self.struct_feats,
                                    'target': self.labels,
                                    'static_ids': self.static_ids,
                                    'program_ids': self.program_ids,
                                   }
                  )

    def load(self):
        print("Loading data ...")
        info_dict = load_info(self.save_path)
        self.code_feats = info_dict['code']
        self.struct_feats = info_dict['struct']
        self.labels = info_dict['target']
        self.static_ids = info_dict['static_ids']
        self.program_ids = info_dict['program_ids']

    def has_cache(self):
        if os.path.exists(self.save_path):
            print("Data exists")
            return True
        return False



if __name__ == '__main__':
    config = read_config_file("config/wala.config")
    data = FinetunedDataset(config, "train")
    data = FinetunedDataset(config, "test")
