from tqdm import tqdm
import numpy as np
from src.finetune.dataset import CallGraphDataset
from torch.utils.data import DataLoader
from src.finetune.model import BERT
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torch
import argparse
from src.utils.utils import Logger, AverageMeter, evaluation_metrics, read_config_file
import os
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PARAMS = {'batch_size': 15, 'shuffle': True, 'num_workers': 8}
TEST_PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8}

logger = Logger()

def train(dataloader, model, mean_loss, loss_fn, optimizer, cfx_matrix):
    model.train()
    loop=tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    for idx, batch in loop:
        code_ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        label = batch['label'].to(device)
        output, _=model(
                ids=code_ids,
                mask=mask)

        loss = loss_fn(output, label)

        num_samples = output.shape[0]
        mean_loss.update(loss.item(), n=num_samples)
        
        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()
        
        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)

        logger.log("Iter {}: Loss {}, Precision {}, Recall {}, F1 {}".format(idx, mean_loss.item(), precision, recall, f1))
        loop.set_postfix(loss=mean_loss.item(), pre=precision, rec=recall, f1 = f1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, cfx_matrix

def do_test(dataloader, model):
    model.eval()
    cfx_matrix = np.array([[0, 0],
                           [0, 0]])
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for batch, dl in loop:
        ids=dl['ids'].to(device)
        mask= dl['mask'].to(device)
        label=dl['label'].to(device)
        output, _=model(
                ids=ids,
                mask=mask)
        
        output = F.softmax(output)
        output = output.detach().cpu().numpy()[:, 1]
        pred = np.where(output >= 0.5, 1, 0)
        label = label.detach().cpu().numpy()
        
        cfx_matrix, precision, recall, f1 = evaluation_metrics(label, pred, cfx_matrix)
        loop.set_postfix(pre=precision, rec=recall, f1 = f1)
        
    (tn, fp), (fn, tp) = cfx_matrix
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = 2*precision*recall/(precision + recall)
    logger.log("[EVAL] Iter {}, Precision {}, Recall {}, F1 {}".format(batch, precision, recall, f1))

def do_train(epochs, train_loader, test_loader, model, loss_fn, optimizer, learned_model_dir):
    cfx_matrix = np.array([[0, 0],
                           [0, 0]])
    mean_loss = AverageMeter()
    for epoch in range(epochs):
        logger.log("Start training at epoch {} ...".format(epoch))
        model, cfx_matrix = train(train_loader, model, mean_loss, loss_fn, optimizer, cfx_matrix)
        
        logger.log("Saving model ...")
        torch.save(model.state_dict(), os.path.join(learned_model_dir, "model_epoch{}.pth".format(epoch)))

        logger.log("Evaluating ...")
        do_test(test_loader, model)
    
    torch.save(model.state_dict(), os.path.join(learned_model_dir, "model.pth"))
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
    print("Running on config {}".format(args.config_path))
    print("Mode: {}".format(args.mode))
    
    mode = args.mode
    learned_model_dir = config["LEARNED_MODEL_DIR"]


    train_dataset= CallGraphDataset(config, "train")
    test_dataset= CallGraphDataset(config, "test")

    print("Dataset have {} train samples and {} test samples".format(len(train_dataset), len(test_dataset)))

    train_loader = DataLoader(train_dataset, **TRAIN_PARAMS)
    test_loader = DataLoader(test_dataset, **TEST_PARAMS)

    model=BERT()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer= optim.Adam(model.parameters(),lr= 0.00001)

    if mode == "train":
        do_train(2, train_loader, test_loader, model, loss_fn, optimizer, learned_model_dir)
    elif mode == "test":
        model.load_state_dict(torch.load(args.model_path))
        do_test(test_loader, model)
    else:
        raise NotImplemented

if __name__ == '__main__':
    main()
