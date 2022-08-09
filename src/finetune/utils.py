import torch
import os
from tqdm import tqdm
from src.finetune.dataset import CallGraphDataset
from torch.utils.data import DataLoader
from src.finetune.model import BERT
import numpy as np
from torch import nn

PARAMS = {'batch_size': 10, 'shuffle': False, 'num_workers': 8} 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_finetune(config, mode):
    dataset= CallGraphDataset(config, mode)
    dataloader = DataLoader(dataset, **PARAMS)
    model_path = os.path.join(config["LEARNED_MODEL_DIR"], "model.pth")
    save_dir = os.path.join(config["CACHE_DIR"], "{}_finetuned".format(mode))
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    model=BERT()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    loop=tqdm(enumerate(dataloader),leave=False,total=len(dataloader))
    for idx, batch in loop:
        ids=batch['ids'].to(device)
        mask= batch['mask'].to(device)
        _, emb =model(
                ids=ids,
                mask=mask)
        emb = emb.detach().cpu().numpy()
        save_path = os.path.join(save_dir, "{}.npy".format(idx))
        np.save(save_path, emb)
    


