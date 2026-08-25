from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler, random_split, Subset
from torch.utils.data.sampler import RandomSampler
import pandas as pd
from dateutil.relativedelta import relativedelta
import pandas as pd
import gc
from sklearn.model_selection import TimeSeriesSplit


def get_automated_date_map(test_end_str):
    t_end = pd.to_datetime(test_end_str)
    t_start = (t_end - relativedelta(years=9, months=11)).replace(day=1)
    tr_end = t_start - pd.Timedelta(days=1)
    tr_start = (tr_end - relativedelta(years=14, months=11)).replace(day=1)

    return {
        'train': (tr_start.strftime('%Y-%m-%d'), tr_end.strftime('%Y-%m-%d')),
        'test': (t_start.strftime('%Y-%m-%d'), t_end.strftime('%Y-%m-%d'))
    }



class RobustZScoreNorm():
    def __init__(self, fit_start_time=None, fit_end_time=None, fields_group=None, clip_outlier=True, mode='daily'):
        """
        Normalization for Financial Time Series
        mode: 
          - 'daily': Cross-sectional normalization (Stock features)
          - 'global': Time-series normalization (Market features) for benchmark models using market features
        """
        self.fit_start_time = fit_start_time
        self.fit_end_time = fit_end_time
        self.fields_group = fields_group
        self.clip_outlier = clip_outlier
        self.mode = mode 
        self.daily_medians = None # Daily normalization
        self.daily_stds = None 
        self.mean_train = None # Global normalization
        self.std_train = None

    def fit(self, df: pd.DataFrame):
        # 1. Daily Mode: Cross-sectional Normalization
        if self.mode == 'daily':
            # Cross sectional Normalization 
            valid_df = df[df['is_scaling'] == True]
            group_level = 'date'
            # Create the groupby object once
            grouped = valid_df.groupby(level=group_level)[self.fields_group]
            self.daily_medians = grouped.median()
            medians_expanded = grouped.transform('median') # Expanding dimension of Date
            abs_deviation = (valid_df[self.fields_group] - medians_expanded).abs()
            self.daily_stds = abs_deviation.groupby(level=group_level).median() * 1.4826 + 1e-12


        # 2. Global Mode: Time-dependent Normalization
        else:
            train_df = df.loc[self.fit_start_time : self.fit_end_time]
            X = train_df[self.fields_group].values
            self.mean_train = np.median(X, axis=0)
            self.std_train = np.median(np.abs(X - self.mean_train), axis=0) * 1.4826 + 1e-12

    def __call__(self, df):
        df = df.copy()
        group_level = 0 if isinstance(df.index, pd.MultiIndex) else df.index.name
        # print(self.daily_medians)
        if self.mode == 'daily':
            m = self.daily_medians.reindex(df.index.get_level_values(group_level)).values
            s = self.daily_stds.reindex(df.index.get_level_values(group_level)).values
            df[self.fields_group] = (df[self.fields_group].values - m) / s
            
        else:
            X = df[self.fields_group].values
            df[self.fields_group] = (X - self.mean_train) / self.std_train
        
        if self.clip_outlier:
            df[self.fields_group] = df[self.fields_group].clip(-3, 3)
            
        return df


class Dataset(Dataset):
    def __init__(self, df, market_df, flag='train', seq_len=8, target='LABEL', test_date = '2025-12-31'):
        self.seq_len = seq_len
        self.flag = flag
        self.target = target

        self.feature_cols = [
                    # 1. Momentum Indicators (Oscillators)
                    'RSI', 'WilliamsR', 'ROC', 'PPO', 'MOM', 'CCI',
                    'Stoch_K', 'Stoch_D',

                    # 2. Trend Indicators
                    'SMA', 'EMA', 'WMA', 'SAR', 'ADX',
                    'MACD', 'MACD_Signal', 'MACD_Hist',

                    # 3. Volatility Indicators
                    'ATR', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Percent',

                    # 4. Volume Indicators
                    'MFI', 'OBV', 'ADOSC',

                    # 5. Price Returns & Ratios (Log-transformed)
                    'return', 'open_r', 'high_r', 'low_r']

        self.tradable_col = 'is_tradable'
        
        self.rel_cols = ['ff12_idx']
        # 2. INDEX
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        market_df['date'] = pd.to_datetime(market_df['date'])
        self.market_df = market_df.set_index('date').sort_index()
        self.market_cols = list(self.market_df.columns) # Features
        
        # 3. All Period
        self.all_dates = sorted(df['date'].unique())
        self.permnos = sorted(df['permno'].unique())
        print(len(self.permnos))

        
        # tradable for seqeunce length : Tradable & seqeunce length all exist
        full_index = pd.MultiIndex.from_product([self.all_dates, self.permnos], names=['date', 'permno'])
        df = df.set_index(['date', 'permno']).reindex(full_index)
        for col in self.feature_cols + self.rel_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['has_data'] = df[self.feature_cols + self.rel_cols].notna().all(axis=1)
        df['data_available'] = df.groupby('permno')['has_data'].transform(
            lambda x: x.rolling(window=self.seq_len).min()
        ).fillna(0).astype(bool)
        df['is_tradable'] = (df['is_tradable'].fillna(0) == 1) & (df['data_available'])
        df['is_tradable'] = df['is_tradable'].astype(bool)

        df['is_scaling'] = df['is_tradable'].copy() # for crossectional scaler
        
        df[self.feature_cols + self.rel_cols] = df[self.feature_cols + self.rel_cols].astype(np.float32)

        self.df = df
        
        date_map = get_automated_date_map(test_date)
        print(f"\n{'='*60}")
        print(f"  [TRAIN AND VAL]      : {date_map['train'][0]} ~ {date_map['train'][1]}")
        print(f"  [TEST]       : {date_map['test'][0]} ~ {date_map['test'][1]}")
        print(f"{'='*60}\n")
        self.train_start, self.train_end = date_map['train']
        target_start, target_end = date_map[flag]
        
        print("Original is_tradable Count:", self.df.loc[self.train_start:self.train_end]['is_tradable'].sum())
        
        self.target_dates = [d for d in self.all_dates if pd.Timestamp(target_start) <= d <= pd.Timestamp(target_end)]
        
        # Align input dates for sequence lookback
        first_date_idx = self.all_dates.index(self.target_dates[0])
        input_start_idx = max(0, first_date_idx - (seq_len - 1))
        input_end_idx =self.all_dates.index(self.target_dates[-1])
        self.input_dates = self.all_dates[input_start_idx : input_end_idx + 1]

        self.__preprocess__()

    def __preprocess__(self):
        # [Step 1] Normalization for Stock
        self.stock_norm = RobustZScoreNorm(fields_group=self.feature_cols, mode='daily')
        self.stock_norm.fit(self.df) # all period (Cross sectional normalization)
        input_stock_df = self.df.loc[self.input_dates[0] : self.input_dates[-1]].copy()
        print("input_stock_scale", input_stock_df.shape)
        processed_stock_df = self.stock_norm(input_stock_df)
        
        # [Step 2] Normalization for Market Index
        self.market_norm = RobustZScoreNorm(
            fit_start_time=self.train_start, 
            fit_end_time=self.train_end, 
            fields_group=self.market_cols, 
            mode='global'
        )
        self.market_norm.fit(self.market_df)
        input_m_df = self.market_df.loc[self.input_dates[0] : self.input_dates[-1]].copy()
        processed_m_df = self.market_norm(input_m_df)

        # Reshape stock data into (T, N, F) arrays
        full_index = pd.MultiIndex.from_product([self.input_dates, self.permnos], names=['date', 'permno'])
        df_re = processed_stock_df.reindex(full_index).sort_values(['date', 'permno'])
        T, N = len(self.input_dates), len(self.permnos)
        self.x_array = df_re[self.feature_cols].values.reshape(T, N, len(self.feature_cols)).astype(np.float32)
        self.y_array = df_re[self.target].values.reshape(T, N).astype(np.float32)
        self.tradable = df_re[self.tradable_col].values.reshape(T, N).astype(bool)
        self.relation_matrix = df_re[self.rel_cols].values.reshape(T, N, len(self.rel_cols)).astype(np.float32)

        # Reshape market data into (T, M_F) arrays
        self.m_array = processed_m_df.loc[self.input_dates, self.market_cols].values


    def __len__(self):
        return len(self.input_dates) - self.seq_len + 1
        
    def __getitem__(self, index):
        # (T, N, F)
        # All permnos for the given date index and Select only tradable stocks 
        x_slice = self.x_array[index : index + self.seq_len]
        m_slice = self.m_array[index : index + self.seq_len]
        rel_slice = self.relation_matrix[index + self.seq_len - 1]
        y_slice = self.y_array[index + self.seq_len - 1]
        tradable = self.tradable[index + self.seq_len - 1]
        if np.isnan(x_slice[:, tradable]).any():
            print(f"Index {index}: NaN detected in final_x!")
        return (x_slice[:, tradable].transpose(1, 0, 2).astype(np.float32),           # (500, T, F) 
                m_slice.astype(np.float32),                                           # (T, 63) 
                rel_slice[tradable].astype(np.float32),                               # (500, 3) 
                y_slice[tradable].astype(np.float32))                                 # (500, 1) 



class Dataset_all(object):
    def __init__(self, configs):
        self.configs = configs
        
        # Date synchronize 
        df= pd.read_parquet(os.path.join(configs.root_path, 'stock_dataset_final.parquet')) 
        market_df = pd.read_parquet(os.path.join(configs.root_path, 'stock_market.parquet'))
        start = max(df.date.min(), market_df.date.min())
        df = df[df.date >= start]
        market_df = market_df[market_df.date >= start]
        df = df.sort_values(['date', 'permno']).reset_index(drop=True)
        market_df = market_df.sort_values('date').reset_index(drop=True)
        
        full_set = Dataset(df, market_df, flag='train', seq_len=configs.seq_len, test_date = configs.test_date)

        total_indices = np.arange(len(full_set))
        num_blocks = 10
        block_indices = np.array_split(total_indices, num_blocks)
        
        train_list = []
        val_list = []
        gap = configs.seq_len
        for i in range(num_blocks):
            if (i + 1) % 2 == 0: 
                val_list.append(block_indices[i])
            else:          
                purged_block = block_indices[i][gap:-gap] 
                train_list.append(purged_block)
        
        self.train_indices = np.concatenate(train_list)
        self.val_indices = np.concatenate(val_list)
        
        self.train_set = Subset(full_set, self.train_indices)
        self.val_set = Subset(full_set, self.val_indices)
        self.test_set = Dataset(df, market_df, flag='test', seq_len=configs.seq_len, test_date = configs.test_date)
        # Memory clear
        del df
        del market_df
        gc.collect()


        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=1, 
            sampler=RandomSampler(self.train_set), 
            num_workers=configs.num_workers,
            pin_memory=True # GPU Speed
        )
        
        self.validation_loader = DataLoader(
            self.val_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=configs.num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=1, 
            shuffle=False, 
            num_workers=configs.num_workers,
            pin_memory=True
        )

    def _get_data(self, flag='train'):
        if flag == 'train':
            return self.train_set, self.train_loader
        elif flag == 'validation':
            return self.val_set, self.validation_loader
        elif flag == 'test':
            return self.test_set, self.test_loader
        else:
            raise ValueError(f"Invalid flag: {flag}")