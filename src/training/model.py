from torch import nn
import torch
class NNClassifier_Combine(nn.Module):
    def __init__(self, hidden_size = 16):
        super(NNClassifier_Combine, self).__init__()
        self.encoder1 = nn.Linear(768, hidden_size)
        self.encoder2 = nn.Linear(22, hidden_size)
        self.decoder = nn.Linear(2 * hidden_size, 2)

    def forward(self, code, struct):
        h_c = self.encoder1(code)
        h_s = self.encoder2(struct)
        h = torch.cat([h_c, h_s], axis=1)
        out = self.decoder(h)  
        return out  


class NNClassifier_Semantic(nn.Module):
    def __init__(self, hidden_size = 16):
        super(NNClassifier_Semantic, self).__init__()
        self.encoder1 = nn.Linear(768, hidden_size)
        self.encoder2 = nn.Linear(22, hidden_size)
        self.decoder = nn.Linear(2 * hidden_size, 2)
        self.decoder2 = nn.Linear(hidden_size, 2)
    
    
    def forward(self, code, struct):
        h_s = self.encoder1(code)
        h = h_s
        out = self.decoder2(h)
        return out

class NNClassifier_Structure(nn.Module):
    def __init__(self, hidden_size = 16):
        super(NNClassifier_Structure, self).__init__()
        self.encoder1 = nn.Linear(768, hidden_size)
        self.encoder2 = nn.Linear(22, hidden_size)
        self.decoder = nn.Linear(2 * hidden_size, 2)
        self.decoder2 = nn.Linear(hidden_size, 2)
     
    def forward(self, code, struct):
        h_s = self.encoder2(struct)
        h = h_s
        out = self.decoder2(h)
        return out

