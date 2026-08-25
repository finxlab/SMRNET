#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
import os
import joblib
import argparse
import logging
import os

import numpy as np
from numpy.linalg import inv
from scipy.stats import chi2
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm

# import lib
from lib.datasetLoader import *
from lib.Model import *
from lib.modules import *
from lib.Metric import *

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
logger = logging.getLogger('MODEL.Train')

def evaluate(model, test_loader, configs):
    
    model.eval()
    K_list = [10, 30, 50]  

    with torch.no_grad():
        summary_metric = {}
        eval_dict = {
            'N': 0,     # Total N
            'SE': [],    # Squared Error Sum (A)
            'IC': [],    # 
            'RIC': [],    # 
            'port_rets': {k: [] for k in K_list},
            'bottom_rets': {k: [] for k in K_list}
        }
        for i, (batch_x, batch_m, batch_rel, batch_y) in enumerate(tqdm(test_loader)):
            # print(batch_y)
            
            batch_x = batch_x[0].float().to(configs.device) # input time series
            batch_m = batch_m[0].float().to(configs.device) # market status
            batch_rel = batch_rel[0].int().to(configs.device) # sector information
            batch_y = batch_y[0].float().to(configs.device) # label
            
            N, = batch_y.shape # N : Num of stocks

            outputs = model(batch_x, batch_m, batch_rel)

            outputs = outputs.detach().cpu().numpy().flatten()
            batch_y = batch_y.detach().cpu().numpy().flatten()
            # print(batch_y)
            for k in K_list:
                top_indices = np.argpartition(outputs, -k)[-k:]
                daily_ret = np.mean(batch_y[top_indices])
                eval_dict['port_rets'][k].append(daily_ret)


                bottom_indices = np.argpartition(outputs, k)[:k]
                bottom_ret = np.mean(batch_y[bottom_indices])
                eval_dict['bottom_rets'][k].append(bottom_ret)



            label_z = cszscore(batch_y, clip = False)
            se_val, n_val = MSE(outputs, label_z)
            eval_dict['SE'].append(se_val)
            eval_dict['N'] += n_val
            ic, ric = calc_ic(outputs, label_z) # No scaling
            eval_dict['IC'].append(ic)
            eval_dict['RIC'].append(ric)



    avg_ic = np.mean(eval_dict['IC'])
    std_ic = np.std(eval_dict['IC'])
    avg_ric = np.mean(eval_dict['RIC'])
    std_ric = np.std(eval_dict['RIC'])
    
    
    summary_metric = {
        'Z-MSE': float(np.sum(eval_dict['SE']) / eval_dict['N']),
        'IC': float(avg_ic),
        'IC_IR': float(avg_ic / std_ic) if std_ic > 0 else 0.0,
        'RIC': float(avg_ric),
        'RIC_IR': float(avg_ric / std_ric) if std_ric > 0 else 0.0,
        
    }
    ## PORTFOLIO ##
    predict_len = 20
    ann_factor = 252 / predict_len
    for k in K_list:
        # --- 1. Top Portfolio ---
        top_rets = np.array(eval_dict['port_rets'][k])[::predict_len]
        bottom_rets = np.array(eval_dict['bottom_rets'][k])[::predict_len]

        n_obs = len(top_rets)
        ar_top = np.mean(top_rets) * ann_factor
        ar_bot = np.mean(bottom_rets) * ann_factor
        mean_top = np.mean(top_rets) * ann_factor        # top_r -> top_rets
        std_top  = np.std(top_rets, ddof=1) * np.sqrt(ann_factor)  # top_r -> top_rets
        sr_top   = mean_top / std_top if std_top > 0 else 0.0
        summary_metric[f'AR_Top_{k}'] = float(ar_top)
        summary_metric[f'SR_Top_{k}'] = float(sr_top)
        summary_metric[f'AR_Bot_{k}'] = float(ar_bot)
        
        print(f"K={k:<2} | Top AR: {ar_top:>7.2%} | Bottom AR: {ar_bot:>7.2%} |SR: {sr_top:.2f}")
    print("-" * 80)
    print(f"[Summary] Z-MSE: {summary_metric['Z-MSE']:.6f}")
    print(f"[All Data] IC: {summary_metric['IC']:.6f} | IC_IR: {summary_metric['IC_IR']:.4f} | RIC: {summary_metric['RIC']:.6f} | RIC_IR: {summary_metric['RIC_IR']:.4f}")
    print("-" * 80)
    return summary_metric

def modelTrain(model: nn.Module,
          optimizer: optim,
          train_loader: DataLoader,
          configs)  :

    model.train()
    loss_epoch = np.zeros(len(train_loader))
    criterion1 = nn.MSELoss()

    for i, (batch_x, batch_m, batch_rel, batch_y) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_x = batch_x[0].float().to(configs.device) # input time series
        batch_m = batch_m[0].float().to(configs.device) # market status
        batch_rel = batch_rel[0].int().to(configs.device) # sector information
        batch_y = batch_y[0].float().to(configs.device) # label
        N, = batch_y.shape # N : Num of stocks
        label_z = cszscore(batch_y, clip = True)


        outputs = model(batch_x, batch_m, batch_rel)
        loss = criterion1(outputs, label_z)
        if i % 1000 == 0 :
            print(f"Loss: {loss.item():.5f}")
            print(outputs[:5], batch_y[:5])

        loss.backward()

        # cliping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_epoch[i] = loss.item()

    return loss_epoch




def train_and_evaluate(model: nn.Module,
                       dataset_all: object,
                       optimizer: optim, 
                       configs) :

    logger.info('begin training and evaluation')
    
    # dataset
    train_set, train_loader = dataset_all._get_data('train')
    val_set, validation_loader = dataset_all._get_data('validation')
    test_set, test_loader = dataset_all._get_data('test')


    path = os.path.join(configs.model_dir, configs.setting)

    if not os.path.exists(path):
        os.makedirs(path)

    train_len = len(train_loader)

    loss = configs.loss
    evaluation_summary = {}
    loss_summary = np.zeros((train_len * configs.train_epochs))
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=configs.train_epochs)


    for epoch in tqdm(range(configs.train_epochs)):
        logger.info('Epoch {}/{}'.format(epoch + 1, configs.train_epochs))

        loss_summary[epoch * train_len:(epoch + 1) * train_len] = modelTrain(model, optimizer, train_loader, configs)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch+1}. Adjust lr: {current_lr:.8f}')


        summary_val = evaluate(model, validation_loader, configs)
        summary_test = evaluate(model, test_loader, configs)

        evaluation_summary[epoch] = summary_val
        val_loss = - summary_val[loss] # IC is score
        early_stopping(val_loss, model, path)


        if early_stopping.early_stop:
            print("Early stopping")
            break



    logger.info(f'Current Best Loss is:  {early_stopping.best_score}')

    evaluation_summary['best_score'] = early_stopping.best_score
    json_path = os.path.join(path, 'validation_metric.json')
    save_dict_to_json(evaluation_summary, json_path)

    configs_dict = vars(configs)
    configs_dict['device'] = str(configs_dict['device'])
    configs_dict['best score'] = early_stopping.best_score
    json_path = os.path.join(path, 'configs.json')
    save_dict_to_json(configs_dict, json_path)

