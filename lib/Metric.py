import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def calc_ic_pd(pred, label):

    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')

    return ic, ric

def calc_ic(pred, label):
    # Pearson IC
    ic = np.corrcoef(pred, label)[0, 1]
    # Rank IC (Spearman)
    ric, _ = spearmanr(pred, label)
    
    return ic, ric

def MSE(prediction_samples, labels):
    N, = labels.shape
    return np.sum((prediction_samples - labels) ** 2), N 
