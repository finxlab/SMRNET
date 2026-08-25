from torch.nn.utils import parametrizations
from typing import Optional, Tuple
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import json
import shutil
from lib.modules import * # for basic module
logger = logging.getLogger('SMRNET.Model') 
import math


class UnifiedModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.model_type = configs.model_type
        if self.model_type == "SMRNET":
            self.model = SMRNET(configs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def forward(self, x, m, r):
        # m is for benchmark models using market features (not used in the proposed model)
        if self.model_type == "SMRNET" :
            output = self.model(x, m, r)
        return output


class orthogonal_cayley(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        self.bias= nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        X = self.weight
        X = X.tril(diagonal=-1)
        A = X - X.mT
        Id = torch.eye(self.dim, dtype=A.dtype, device=A.device)
        left = torch.add(Id, A, alpha=-0.5)
        right =  torch.add(Id, A, alpha=0.5)
        Q = torch.linalg.solve(left, right)
        return nn.functional.linear(input, Q, None) 


def find_sector(x, r):
    N, D = x.shape
    sum_val = torch.matmul(r.squeeze(), x) # N D

    count = (r != 0).sum(axis =1)
    sector_inform = sum_val / count
    return sector_inform



class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias= nn.Parameter(torch.Tensor(out_features))
        
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)
    def forward(self, input):
        return nn.functional.linear(input, torch.square(self.weight), None)



class SMRNET(nn.Module):
    def __init__(self, configs):
        super(SMRNET, self).__init__()
        self.configs = configs
        self.stock_embedding = StockCNNEmbedding(input_size=configs.feature_dim, hidden_size=configs.hidden_dim, dropout = configs.dropout)
        self.stock_context = TemporalSelfAttention(configs.hidden_dim, configs.n_heads, dropout=configs.dropout)

        self.feature_extractor = orthogonal_cayley(configs.hidden_dim)
        self.feature_gat = orthogonal_cayley(configs.hidden_dim)
        self.feature_sector = orthogonal_cayley(configs.hidden_dim)
        
        self.GNN = MarketGAT(hidden_dim = configs.hidden_dim, GAT_heads = configs.GAT_heads, dropout = configs.dropout, rel_dim = configs.rel_dim)

        self.SIF = SignedInteractionFusion(configs.hidden_dim, configs.dropout)
        self.final_linear = PositiveLinear(configs.hidden_dim * 2 , 1) # Residual path
    def forward(self, x, m, r):
        """
        N : num of stocks
        x: [N, T, F]
        m: [T, Fm] is for benchmark models using market features (not used in the proposed model)
        rel_matrix: [N, 1]
        """
        N = x.size(0)

        x_emb = self.stock_embedding(x)
        rel_matrix = (r[:, None, :] == r[None, :, :]).float()

        stock_inform = self.feature_extractor(self.stock_context(x_emb))
        sector_inform = find_sector(stock_inform, rel_matrix)

        gat, sector = self.GNN(stock_inform, sector_inform, rel_matrix)
        gat = self.feature_gat(gat)  + stock_inform
        sector = self.feature_sector(sector)  + stock_inform

        fused = self.SIF(gat, sector)
        pred =  self.final_linear(fused)
        self.fused = fused
        return pred.squeeze(-1)




class SignedInteractionFusion(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.adaptive_scale = nn.Parameter(torch.zeros(1))
        self.dim_scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x1, x2):
        pos1, neg1 = torch.relu(x1), torch.relu(-x1)
        pos2, neg2 = torch.relu(x2), torch.relu(-x2)
        pos1, pos2 = self.dropout(pos1), self.dropout(pos2)
        neg1, neg2 = self.dropout(neg1), self.dropout(neg2)
        
        inter_plus = (pos1 * pos2) / self.dim_scale
        inter_minus = (neg1 * neg2) / self.dim_scale
        fused_plus = pos1 + pos2 + inter_plus
        fused_minus = -(neg1 + neg2 + inter_minus)
        fused = torch.cat([fused_plus, fused_minus], dim=-1)
        return fused / (self.adaptive_scale.abs() + 1)




class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]



class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.pos_emb = PositionalEmbedding(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout = dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.norm(x + self.pos_emb(x))
        attn_out, _ = self.mha(x[:, -1:], x, x)
        return attn_out.squeeze(1)


class StockCNNEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        out_channels = hidden_size // 3
        self.conv1 = nn.Conv1d(input_size, out_channels, kernel_size=1)
        self.conv5 = nn.Conv1d(input_size, out_channels, kernel_size=5, padding=2)
        self.conv11 = nn.Conv1d(input_size, hidden_size - (out_channels * 2), kernel_size=11, padding=5)
        self.ln = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        c1 = self.conv1(x)
        c5 = self.conv5(x)
        c11 = self.conv11(x)
        out = torch.cat([c1, c5, c11], dim=1)
        out = self.activation(out)
        out = self.dropout(out)
        return out.transpose(1, 2)




class Neutralizer(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.activation = nn.LeakyReLU(0.01) 

    def forward(self, x, sector_inform):

        dot_product = torch.sum(x * sector_inform, dim=-1, keepdim=True)    
        sector_norm_sq = torch.sum(sector_inform * sector_inform, dim=-1, keepdim=True) + self.eps
        beta = dot_product / sector_norm_sq        
        BETA = beta * sector_inform  
        ALPHA = x - BETA
        return BETA, ALPHA


class MarketGAT(nn.Module):
    def __init__(self, hidden_dim, GAT_heads, dropout, rel_dim):
        super(MarketGAT, self).__init__()
        self.rel_dim = rel_dim
        self.a_i_alpha = nn.Parameter(torch.Tensor(hidden_dim, GAT_heads))
        self.a_j_alpha = nn.Parameter(torch.Tensor(hidden_dim, GAT_heads))

        self.a_i_beta = nn.Parameter(torch.Tensor(hidden_dim, GAT_heads))
        self.a_j_beta = nn.Parameter(torch.Tensor(hidden_dim, GAT_heads))
        self.dropout = nn.Dropout(dropout)
        self.sector_neutral = Neutralizer()
        nn.init.xavier_uniform_(self.a_i_alpha); nn.init.xavier_uniform_(self.a_j_alpha)
        nn.init.xavier_uniform_(self.a_i_beta); nn.init.xavier_uniform_(self.a_j_beta)

    def forward(self, x, sector_inform, r):

        """
        x: [N, F], m: [1, F], r: [N, N, 3]
        """
        BETA, ALPHA = self.sector_neutral(x, sector_inform)

        # --- Stream 1: ALPHA GAT ---
        f_i_a = torch.matmul(ALPHA, self.a_i_alpha)
        f_j_a = torch.matmul(ALPHA, self.a_j_alpha)
        e_a = f_i_a.T.unsqueeze(-1) + f_j_a.T.unsqueeze(1)

        # --- Stream 2: BETA GAT ---
        f_i_b = torch.matmul(BETA, self.a_i_beta)
        f_j_b = torch.matmul(BETA, self.a_j_beta)
        e_b = f_i_b.T.unsqueeze(-1) + f_j_b.T.unsqueeze(1)
        mask = (r.sum(dim=-1) == 0).unsqueeze(0)

        e_a = e_a.masked_fill(mask, -1e8)
        e_b = e_b.masked_fill(mask, -1e8)
        alpha_a = self.dropout(torch.softmax(F.leaky_relu(e_a, 0.01), dim=-1))
        alpha_b = self.dropout(torch.softmax(F.leaky_relu(e_b, 0.01), dim=-1))

        out_alpha = torch.matmul(alpha_a, ALPHA).mean(0)
        out_beta = torch.matmul(alpha_b, BETA).mean(0)

        return out_alpha , out_beta



def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    
    
    if is_best:
        torch.save(state, os.path.join(checkpoint, f'ins_{ins_name}_best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')

def load_checkpoint(checkpoint, model, optimizer = None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        optimizer_dain: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
        
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
        
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        optimizer.param_groups[0]['capturable'] = True

    return checkpoint

    
def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_json(path) :
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


