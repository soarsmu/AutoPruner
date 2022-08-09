import dgl
import dgl.function as dgl_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.gnn.dataset import CallGraphDataset
from src.utils.utils import read_config_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

node_feats_name = [
"-trans#depth_from_main",
"-direct#src_node_in_deg",
"-direct#src_node_out_deg",
"-trans#src_node_in_deg",
"-trans#src_node_out_deg",
"-direct#dest_node_out_deg",
"-direct#dest_node_in_deg",
"-trans#dest_node_out_deg",
"-trans#dest_node_in_deg",
]

def prepare_header(tool_header):
    new_node_feats_name = []
    new_edge_feats_name = []
    for header in node_feats_name:
        if header != "wiretap":
            header = tool_header + header
        new_node_feats_name.append(header)
    
    for header in edge_feats_name:
        if header != "wiretap":
            header = tool_header + header
        new_edge_feats_name.append(header)

    return new_node_feats_name, new_edge_feats_name

class GCNLayer(nn.Module):
    def __init__(self, hidden_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(hidden_features, hidden_features)
        self.relu = nn.ReLU(hidden_features)

    def forward(self, g):
        g.ndata['h'] = self.linear(g.ndata['h'])
        g.update_all(dgl_fn.copy_u(u='h', out='m'), dgl_fn.mean(msg='m', out='h'))
        g.ndata['h'] = self.relu(g.ndata['h'])
        return g

class GCNModel(nn.Module):
    def __init__(self, config, hidden_feats, n_layers = 10, num_classes= 2, n_in_feats = len(node_feats_name), e_in_feats = len(edge_feats_name)):
        super(GCNModel, self).__init__()
        self.config = config
        self.node_feats_name, self.edge_feats_name = prepare_header(config["HEADERS"])
        self.encoder = nn.Linear(n_in_feats, hidden_feats)
        self.relu = nn.ReLU()
        self.n_layers = n_layers
        self.h_processes = nn.ModuleList([GCNLayer(hidden_feats) for i in range(self.n_layers)])
        self.decoder = nn.Linear(2 * hidden_feats + e_in_feats , num_classes)
        if num_classes > 1:
            self.last_act = nn.Softmax(dim=1)
        else:
            self.last_act = nn.Sigmoid()
    
    def decode_edge_func(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        n_edges = len(edges)
        feats = torch.cat([h_u, h_v], axis = 1)
        for feat_name in self.edge_feats_name:
            feats = torch.cat([feats, edges.data[feat_name].view(n_edges, 1)], axis = 1)
        
        feats = self.decoder(feats)
        
        return {
                'logits': feats,
                'prob': self.last_act(feats)
        }

    def forward(self, g):
        n_nodes = g.num_nodes()
        for feat_name in self.node_feats_name:
            try:
                g.ndata['h'] = torch.cat((g.ndata['h'], g.ndata[feat_name].view(n_nodes, 1)), axis=1)
            except:
                g.ndata['h'] =  g.ndata[feat_name].view(n_nodes, 1)

        g.ndata['h'] = self.encoder(g.ndata['h'])
        g.ndata['h'] = self.relu(g.ndata['h'])
        
        for i in range(self.n_layers):
            g = self.h_processes[i](g)

        g.apply_edges(self.decode_edge_func)
        return g

if __name__ == '__main__':
    config = read_config_file("config/wala.config")
    data = CallGraphDataset(config, "train")
    graph, lb, sa_lb = data[6]
    model = GCNModel(config, 32, 1)
    g = model(graph)
    print(g.edata['prob'])
