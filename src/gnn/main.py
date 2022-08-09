from tqdm import tqdm
import numpy as np
from src.gnn.dataset import CallGraphDataset
from src.gnn.model import GCNModel
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import argparse
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
import os
import warnings
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve
import torch
import math
# Import statistics Library
import statistics

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {'batch_size': 1, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 1, 'shuffle': False, 'num_workers': 8}

logger = Logger()

def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    model.train()
    loop=tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        g, lb, sa_lb = batch
        g = g.to(device)
        lb = lb.to(device)
        sa_lb = sa_lb
        g = model(g)
        # print(g.edata['logits'])
        loss = loss_fn(g.edata['logits'], lb)
        
        output = g.edata['prob']
        
        num_samples = output.shape[0]
        mean_loss.update(loss.item(), n=num_samples)
        
        
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        lb = lb.detach().cpu().numpy()
        
        cfx_matrix, precision, recall, f1 = evaluation_metrics(lb, pred, cfx_matrix)

        # logger.log("Iter {}: Loss {}, Precision {}, Recall {}, F1 {}".format(idx, loss.item(), precision, recall, f1))
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1 = f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, cfx_matrix

def do_test(dataloader, model, best_f1=None):
    model.eval()
    all_f1 = []
    all_precision = []
    all_recall = []
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for idx, batch in loop:
        g, lb, sa_lb = batch
        g = g.to(device)
        lb = lb.to(device)
        sa_lb = sa_lb
        g = model(g)

        output = g.edata['prob']
        output = output.detach().cpu().numpy()[:, 1]
        output = output * sa_lb
        pred = np.where(output >= 0.5, 1, 0)
        lb = lb.detach().cpu().numpy()
        
        precision = precision_score(lb, pred, labels = [0, 1])
        if math.isnan(precision):
            precision = 0
        recall = recall_score(lb, pred, labels = [0, 1])

        if precision + recall != 0:
            f1 = 2*precision*recall/(precision+recall)
        else:
            f1 = 0
        all_f1.append(f1)   
        all_precision.append(precision) 
        all_recall.append(recall) 

    logger.log("[EVAL-AVG] Iter {}, Precision {} ({}), Recall {}({}), F1 {}({})".format(idx, statistics.mean(all_precision),
                                                                                            statistics.stdev(all_precision), 
                                                                                            statistics.mean(all_recall),
                                                                                            statistics.stdev(all_recall),
                                                                                            statistics.mean(all_f1),
                                                                                            statistics.stdev(all_f1)))

    if best_f1 != None and statistics.mean(all_f1) > best_f1:
        best_f1 = statistics.mean(all_f1)
        torch.save(model.state_dict(), "gnn_peta.pth")
    
    return best_f1

def do_train(epochs, train_loader, test_loader, model, loss_fn, optimizer):
    cfx_matrix = np.array([[0, 0],
                           [0, 0]])
    mean_loss = AverageMeter()
    best_f1 = 0
    for epoch in range(epochs):
        logger.log("Start training at epoch {} ...".format(epoch))
        model, cfx_matrix = train(train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix)
        
        # logger.log("Saving model ...")

        logger.log("Evaluating ...")
        best_f1 = do_test(test_loader, model, best_f1)
    
    logger.log("Done !!!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/wala.config") 
    parser.add_argument("--model_path", type=str, default="../replication_package/model/finetuned_model/model.pth", help="Path to checkpoint (for test only)") 
    parser.add_argument("--mode", type=str, default="train") 
    
    return parser.parse_args()

def main():
    args = get_args()
    config = read_config_file(args.config_path)
    mode = args.mode

    train_dataset= CallGraphDataset(config,mode="train")
    test_dataset= CallGraphDataset(config,mode="test")

    print("Dataset have {} train samples and {} test samples".format(len(train_dataset), len(test_dataset)))

    model=GCNModel(config, 32)

    model.to(device)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
    
    model.apply(init_weights)
    

    loss_fn = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(),lr= 5e-4)

    if mode == "train":
        do_train(100, train_dataset, test_dataset, model, loss_fn, optimizer)
    elif mode == "test":
        model_path = args.model_path
        model.load_state_dict(torch.load(model_path))
        do_test(test_dataset, model, None)
    else:
        raise NotImplemented
    


if __name__ == '__main__':
    main()

    # a = [0, 1, 0, 0, 1, 0]
    # b = [0, 1, 0, 1, 0, 1]
    # fpr, tpr, thresholds = roc_curve(a, b)
    # print(roc_curve(a, b, pos_label=2))