from calendar import c
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import networkx as nx
import dgl
import torch
from torch import nn
from torch import cuda
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl.data import DGLDataset
from src.utils.utils import read_config_file
from transformers import AutoTokenizer, AutoModel
use_cuda = cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

header_names = [
"wiretap",
"-direct",
"-trans",
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

edge_feats_name = [
"-direct",
"-trans",
"-direct#repeated_edges",
"-direct#fanout",
"-direct#graph_node_count", 
"-direct#graph_edge_count",
"-direct#graph_avg_deg",
"-direct#graph_avg_edge_fanout",
"-trans#repeated_edges",
"-trans#fanout",
"-trans#graph_node_count",
"-trans#graph_edge_count",
"-trans#graph_avg_deg",
"-trans#graph_avg_edge_fanout"
]

caller_feats_name = [
"-trans#depth_from_main",
"-direct#src_node_in_deg",
"-direct#src_node_out_deg",
"-trans#src_node_in_deg",
"-trans#src_node_out_deg",
]

callee_feats_name = [
"-direct#dest_node_out_deg",
"-direct#dest_node_in_deg",
"-trans#dest_node_out_deg",
"-trans#dest_node_in_deg",
]

def prepare_header(tool_header):
    new_header_names = []
    new_edge_feats_name = []
    new_caller_feats_name = []
    new_callee_feats_name = []
    for header in header_names:
        if header != "wiretap":
            header = tool_header + header
        new_header_names.append(header)
    
    for header in edge_feats_name:
        if header != "wiretap":
            header = tool_header + header
        new_edge_feats_name.append(header)
    
    for header in caller_feats_name:
        if header != "wiretap":
            header = tool_header + header
        new_caller_feats_name.append(header)
    
    for header in callee_feats_name:
        if header != "wiretap":
            header = tool_header + header
        new_callee_feats_name.append(header)

    
    return new_header_names, new_edge_feats_name, new_caller_feats_name, new_callee_feats_name

def load_program_lists(program_lists):
    filenames = []
    with open(program_lists, "r") as f:
        for line in f:
            filenames.append(line.strip())
    return filenames

def get_idx2nodes(edges):
    idx2nodes = []
    nodes2idx = {}
    for src, dst in edges:
        if src not in idx2nodes:
            idx2nodes.append(src)
        if dst not in idx2nodes:
            idx2nodes.append(dst)
    for i, node in enumerate(idx2nodes):
        nodes2idx[node] = i
    return idx2nodes, nodes2idx
    
class CallGraphDataset(DGLDataset):
    def __init__(self, config, mode ="train"):
        save_dir = os.path.join(config["GNN_CACHE_DIR"], mode)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.mode = mode
        self.config = config
        if self.mode == "train":
            self.program_lists = os.path.join(self.config["TRAINING_PROGRAMS_LIST"])
        elif self.mode == "test":
            self.program_lists = os.path.join(self.config["TEST_PROGRAMS_LIST"])
        else:
            NotImplemented
        self.header_names, self.edge_feats_name, self.caller_feats_name, self.callee_feats_name = prepare_header(config["HEADERS"])

        self.graphs = []
        self.labels = []
        self.sa_labels = []

        self.graph_save_path = os.path.join(save_dir, 'dgl_graphs.bin')
        self.info_save_path = os.path.join(save_dir, 'info.pkl')
        super(CallGraphDataset, self).__init__(
            name = 'key_value_dataset',
            save_dir=save_dir
        )
    def __getitem__(self, i):
        g, g_lb, sa_lb = self.graphs[i], self.labels[i], self.sa_labels[i]
        return g, torch.tensor(g_lb, dtype=torch.long), np.array(sa_lb)

    def __len__(self):
        return len(self.graphs)

    def process(self):

        filenames = load_program_lists(self.program_lists)
        for i in tqdm(range(len(filenames))):
            #Load data
            fn = filenames[i]
            data_path = os.path.join(self.config["BENCHMARK_CALLGRAPHS"], fn, self.config["FULL_FILE"])
            features = pd.read_csv(data_path)
            features = features.replace(1000000000, 100000)
            all_srcs = features['method']
            all_dsts = features['target']
            

            all_edges = list(zip(all_srcs, all_dsts))
            g_label = features['wiretap'].to_numpy()
            sa_label = features[self.config["SA_LABEL"]]
            
            #mapping nodes name to idx
            idx2nodes, nodes2idx = get_idx2nodes(all_edges)
            num_nodes = len(idx2nodes)
            static_src_ids = []
            static_dst_ids = []
            
            #create graph
            for src, dst in all_edges:
                static_src_ids.append(nodes2idx[src])
                static_dst_ids.append(nodes2idx[dst])
            g = dgl.graph((torch.tensor(static_src_ids), torch.tensor(static_dst_ids)))
            
            #Create label - Get unreachable nodes
            cnt = 0
            
            for feat_name in self.caller_feats_name:
                g.ndata[feat_name] = torch.zeros(num_nodes)
                
            for feat_name in self.callee_feats_name:
                g.ndata[feat_name] = torch.zeros(num_nodes)
            
            curr_idx = 0
            for src, dst in all_edges:
                for feat_name in self.caller_feats_name:
                    g.ndata[feat_name][nodes2idx[src]] = features[feat_name][curr_idx]
                for feat_name in self.callee_feats_name:
                    g.ndata[feat_name][nodes2idx[dst]] = features[feat_name][curr_idx]
            
            for feat_name in self.edge_feats_name:
                g.edata[feat_name] = torch.FloatTensor(features[feat_name])
            
            self.graphs.append(g)
            self.labels.append(g_label)
            self.sa_labels.append(sa_label)

    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print(len(self.graphs))
        save_graphs(self.graph_save_path, self.graphs)
        save_info(self.info_save_path, {
                                    'label': self.labels,
                                    'sa_label': self.sa_labels
                                   }
                  )


    def load(self):
        print("Loading data ...")
        self.graphs, _ = load_graphs(self.graph_save_path)
        info_dict = load_info(self.info_save_path)
        self.labels = info_dict['label']
        self.sa_labels = info_dict['sa_label']

       

    def has_cache(self):
        if os.path.exists(self.graph_save_path) and os.path.exists(self.info_save_path):
            print("Data exists")
            return True
        return False

if __name__ == "__main__":
    config = read_config_file("config/wala.config")
    data = CallGraphDataset(config, mode="test")
    # data = CallGraphDataset(mode="test")
