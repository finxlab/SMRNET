import wrds
import os
import glob
import json
import talib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import gc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--topk', type = int, default=500, help='Number of trading stocks')

class CRSPDATASET:
    def __init__(self, topk=500, start_year=2011, end_year=2025, output_dir='./dataset'):
        """
        :param topk: Number of top market cap stocks to include
        :param start_year: Analysis start year
        :param end_year: Analysis end year
        :param output_dir: Directory to save processed data
        """
        self.topk = topk
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir
        self.warmup_time = 60 # Period required for technical indicator stability
        self.meta_cols = ['permno', 'date', 'is_tradable']
        self.cols = [
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

        self.rel_cols = ['ff5_idx', 'ff12_idx', 'ff48_idx']
        self.target = ['LABEL']
        self.DB_LAST_DATE = None

        self.save_cols = self.meta_cols + self.cols + self.target + self.rel_cols


        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print("Connecting to WRDS DB...")
        self.db = wrds.Connection()
        self.universe_permnos = None

    def get_universe(self):
        """
        Extract top K PERMNOs within the period.
        Targets normal common stocks (NS) listed on Nasdaq, NYSE, or AMEX.
        """
        
        print(f"--- Starting Top {self.topk} PERMNO extraction for {self.start_year}~{self.end_year} ---")
        sql = f"""
        SELECT DISTINCT permno
        FROM (
            SELECT permno, dlycaldt,
                ROW_NUMBER() OVER (PARTITION BY dlycaldt ORDER BY dlycap DESC) AS mcap_rank
            FROM crsp_a_stock.dsf_v2
            WHERE dlycaldt >= '{self.start_year}-01-01' AND dlycaldt <= '{self.end_year}-12-31'
              AND dlycap IS NOT NULL AND sharetype = 'NS' AND primaryexch IN ('N', 'Q', 'A')
        ) AS rank_table
        WHERE mcap_rank <= {self.topk}
        """
        try:
            df = self.db.raw_sql(sql)
            self.universe_permnos = np.sort(df['permno'].astype(int).tolist())
            print(f"Extraction Complete: {len(self.universe_permnos)} unique PERMNOs")
            return self.universe_permnos
        except Exception as e:
            print(f"Error during universe extraction: {e}")
            return []

    def download_data_yearly(self):
        """
        Download yearly stock price data for the identified universe and save as Parquet.
        """
        if self.universe_permnos is None:
            self.get_universe()
            
        permno_str = ",".join(map(str, self.universe_permnos))
        for year in range(self.start_year, self.end_year + 1):
                
            sql = f"""
                SELECT 
                    a.permno, 
                    a.dlycaldt AS date, 
                    a.dlycap AS mktcap, 
                    a.dlyopen AS openprc, 
                    a.dlyhigh AS askhi, 
                    a.dlylow AS bidlo, 
                    a.dlyprc AS prc,
                    a.dlyvol AS vol, 
                    a.dlyret AS ret, 
                    a.dlycumfacpr AS cfacpr, 
                    a.dlycumfacshr AS cfacshr,
                    a.ticker,
                    a.siccd,
                    a.primaryexch,
                    d.delret AS dlret,             
                    a.dlydelflg AS dlstcd         
                FROM crsp_a_stock.dsf_v2 AS a
                LEFT JOIN crsp_a_stock.stkdelists AS d     
                    ON a.permno = d.permno 
                    AND a.dlycaldt = d.delistingdt         
                WHERE a.permno IN ({permno_str})
                AND a.dlycaldt >= '{year}-01-01' 
                AND a.dlycaldt <= '{year}-12-31'
            """
            try:
                df = self.db.raw_sql(sql, date_cols=['date'])
                # 1. Take absolute values for price columns and basic cleaning
                for col in ['openprc', 'askhi', 'bidlo', 'prc'] :
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').abs()
                
                # 2. Handle missing delisting returns
                df['dlret'] = pd.to_numeric(df['dlret'], errors='coerce')
                df['dlret'] = df['dlret'].fillna(0)
                # 3. Save to Parquet
                file_path = os.path.join(self.output_dir, f'crsp_{year}_topN.parquet')
                df.to_parquet(file_path, compression='snappy', index=False)
                print(f"{year} Saving complete: {len(df)} rows")

            except Exception as e:
                print(f"{year} Download/Save Error: {e}")



    def build_market_features(self, benchmark_permnos={'84398': 'Large', '88222': 'Small', '90878': 'Total'}):
        """
        Reconstruct 63 market features using SPY, IWM, and VTI data.
        """
        print("--- Reconstructing Market Time Series (63 Features) ---")
        sql = f"""
            SELECT 
                a.permno, 
                a.ticker,
                a.dlycaldt AS date,           
                a.dlyret AS ret,              
                a.dlyprc AS prc,              
                a.dlyvol AS vol,              
                a.dlycumfacshr AS cfacshr     
            FROM crsp_a_stock.dsf_v2 AS a     
            WHERE a.permno IN ('88222', '84398', '90878')
            AND a.dlycaldt >= '{self.start_year}-01-01' 
            AND a.dlycaldt <= '{self.end_year}-12-31'
            ORDER BY a.dlycaldt
        """
        df = self.db.raw_sql(sql, date_cols=['date'])
        df['ret'] = pd.to_numeric(df['ret'], errors='coerce').fillna(0.0)
        df['prc'] = df['prc'].abs()
        
        all_trading_dates = sorted(df['date'].unique())
        permno_map = {88222: 'Small', 84398: 'Large', 90878: 'Total'}

        def apply_ret_logic(g):
            print(f"Processing ticker: {g['ticker'].iloc[0]}")
            p_id = int(g['permno'].iloc[-1])
            
            # 1. Date normalization (Maintain ascending order)
            g = g.set_index('date').reindex(all_trading_dates).reset_index()
            
            # 2. Restore basic info
            g['permno'] = p_id # Recent value
            g['cfacshr'] = g['cfacshr'].fillna(1.0)
            g['ret'] = g['ret'].fillna(0.0)
            
            # 3. Find the base point (most recent valid price) in ascending order
            last_idx = g['prc'].last_valid_index()
            if last_idx is None: return None
            
            base_price = np.abs(g.loc[last_idx, 'prc'])
            
            # 4. Calculate cumulative growth (ascending)
            growth = (1 + g['ret']).cumprod()
            
            # 5. Calculate adjusted close (Mathematical backward adjustment)
            g['close'] = base_price * (growth / growth.loc[last_idx])
            
            # 6. Calculate trading amount and post-processing
            g['amount'] = g['close'] * (g['vol'] * g['cfacshr'])
            g['close'] = g['close'].ffill()
            g['amount'] = g['amount'].fillna(0)
            
            return g

            

        print("   > Calculating market features...")
        
        groups = []
        for name, group in df.groupby('permno'):
            processed = apply_ret_logic(group)
            if processed is not None:
                groups.append(processed)
        
        raw_df = pd.concat(groups).reset_index(drop=True)
        
        raw_df['layer'] = raw_df['permno'].astype(int).map(permno_map)

        mkt_pivot = raw_df.pivot(index='date', columns='layer', values=['ret', 'amount'])
        mkt_pivot.columns = [f"{layer}_{val}" for val, layer in mkt_pivot.columns]
        intervals = [5, 10, 20, 30, 60]
        features = pd.DataFrame(index=mkt_pivot.index)
        
        for layer in ['Large', 'Small', 'Total']:
            ret_s = mkt_pivot[f"{layer}_ret"].astype(float)
            amt_s = mkt_pivot[f"{layer}_amount"].astype(float).replace(0, np.nan)
            
            features[f"{layer}_ret"] = ret_s
            for d in intervals:
                features[f"{layer}_ret_mean_{d}"] = ret_s.rolling(d).mean()
                features[f"{layer}_ret_std_{d}"] = ret_s.rolling(d).std()
                features[f"{layer}_amt_ratio_{d}"] = amt_s.rolling(d).mean() / amt_s
                features[f"{layer}_amt_std_ratio_{d}"] = amt_s.rolling(d).std() / amt_s

        final_features = features.ffill()
        file_path = os.path.join(self.output_dir, "stock_market.parquet")
        final_features.dropna().reset_index().to_parquet(file_path)
        print(f" Market processing complete! (Total Features: {final_features.shape[1]})")

        return final_features






    def load_universe_data(self):
        """
        Merge all yearly Parquet files into a single DataFrame.
        """
        df_list = []
        for year in range(self.start_year, self.end_year + 1):
            file_path = os.path.join(self.output_dir, f'crsp_{year}_topN.parquet')
            if os.path.exists(file_path):
                print(f"Loading: {file_path}")
                df_list.append(pd.read_parquet(file_path))
            else:
                print(f"File for year {year} not found. (Skipping)")


        if not df_list:
            print("No datasets found to load.")
            return pd.DataFrame()

        full_df = pd.concat(df_list, ignore_index=True).sort_values(['permno', 'date' ]).reset_index(drop=True)
        print(f"--- Load complete : Total {len(full_df)} row ---")
        return full_df
    
    def get_ff5(self, s):
        ff5_map = {
        'Cnsmr': [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), 
                (3100, 3199), (3940, 3989), (2500, 2519), (2590, 2599), 
                (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), 
                (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999), 
                (5000, 5999), (7200, 7299), (7600, 7699)],
        'Manuf': [(2520, 2589), (2600, 2699), (2750, 2769), (2800, 2829), 
                (2840, 2899), (3000, 3099), (3200, 3569), (3580, 3621), 
                (3623, 3629), (3700, 3709), (3712, 3713), (3715, 3715), 
                (3717, 3749), (3752, 3791), (3793, 3799), (3860, 3899), 
                (1200, 1399), (2900, 2999), (4900, 4949)],
        'HiTec': [(3570, 3579), (3622, 3622), (3660, 3692), (3694, 3699), 
                (3810, 3839), (7370, 7372), (7373, 7373), (7374, 7374), 
                (7375, 7375), (7376, 7376), (7377, 7377), (7378, 7378), 
                (7379, 7379), (7391, 7391), (8730, 8734), (4800, 4899)],
        'Hlth': [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]}
        for name, ranges in ff5_map.items():
            for low, high in ranges:
                if low <= s <= high:
                    return name
        return "Other"

    def get_ff12(self, s):
        ff12_map = {
            "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
            "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
            "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
            "Enrgy": [(1200, 1399), (2900, 2999)],
            "Chems": [(2800, 2829), (2840, 2899)],
            "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
            "Telcm": [(4800, 4899)],
            "Utils": [(4900, 4949)],
            "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
            "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
            "Money": [(6000, 6999)]
        }
        for name, ranges in ff12_map.items():
            for low, high in ranges:
                if low <= s <= high:
                    return name
        return "Other"
        
    def get_ff48(self, s):
        ff48_map = {
            "Agric": [(100, 199), (200, 299), (700, 799), (910, 919), (2048, 2048)],
            "Food": [(2000, 2009), (2010, 2019), (2020, 2029), (2030, 2039), (2040, 2046), (2050, 2059), (2060, 2063), (2070, 2079), (2090, 2092), (2095, 2095), (2098, 2099)],
            "Soda": [(2064, 2068), (2086, 2086), (2087, 2087), (2096, 2096), (2097, 2097)],
            "Beer": [(2080, 2080), (2082, 2082), (2083, 2083), (2084, 2084), (2085, 2085)],
            "Smoke": [(2100, 2199)],
            "Toys": [(920, 999), (3650, 3651), (3652, 3652), (3732, 3732), (3930, 3931), (3940, 3949)],
            "Fun": [
                (7800, 7829), (7830, 7833), (7840, 7841), 
                (7900, 7900), (7910, 7911), (7920, 7929), (7930, 7933), 
                (7940, 7949), (7980, 7980), (7990, 7999)
            ],
            "Books": [(2700, 2709), (2710, 2719), (2720, 2729), (2730, 2739), (2740, 2749), (2770, 2771), (2780, 2789), (2790, 2799)],
            "Hshld": [(2047, 2047), (2391, 2392), (2510, 2519), (2590, 2599), (2840, 2843), (2844, 2844), (3160, 3161), (3170, 3171), (3172, 3172), (3190, 3199), (3229, 3229), (3260, 3260), (3262, 3263), (3269, 3269), (3230, 3231), (3630, 3639), (3750, 3751), (3800, 3800), (3860, 3861), (3870, 3873), (3910, 3911), (3914, 3914), (3915, 3915), (3960, 3962), (3991, 3991), (3995, 3995)],
            "Clths": [(2300, 2390), (3020, 3021), (3100, 3111), (3130, 3131), (3140, 3149), (3150, 3151), (3963, 3965)],
            "Hlth": [(8000, 8099)],
            "MedEq": [(3693, 3693), (3840, 3849), (3850, 3851)],
            "Drugs": [(2830, 2830), (2831, 2831), (2833, 2833), (2834, 2834), (2835, 2835), (2836, 2836)],
            "Chems": [(2800, 2809), (2810, 2819), (2820, 2829), (2850, 2859), (2860, 2869), (2870, 2879), (2890, 2899)],
            "Rubbr": [(3031, 3031), (3041, 3041), (3050, 3053), (3060, 3069), (3070, 3079), (3080, 3089), (3090, 3099)],
            "Txtls": [(2200, 2269), (2270, 2279), (2280, 2284), (2290, 2295), (2297, 2297), (2298, 2298), (2299, 2299), (2393, 2395), (2397, 2399)],
            "BldMt": [(800, 899), (2400, 2439), (2450, 2459), (2490, 2499), (2660, 2661), (2950, 2952), (3200, 3200), (3210, 3211), (3240, 3241), (3250, 3259), (3261, 3261), (3264, 3264), (3270, 3275), (3280, 3281), (3290, 3293), (3295, 3299), (3420, 3429), (3430, 3433), (3440, 3441), (3442, 3442), (3446, 3446), (3448, 3448), (3449, 3449), (3450, 3451), (3452, 3452), (3490, 3499), (3996, 3996)],
            "Cnstr": [(1500, 1511), (1520, 1529), (1530, 1539), (1540, 1549), (1600, 1699), (1700, 1799)],
            "Steel": [(3300, 3300), (3310, 3317), (3320, 3325), (3330, 3339), (3340, 3341), (3350, 3357), (3360, 3369), (3370, 3379), (3390, 3399)],
            "FabPr": [(3400, 3400), (3443, 3443), (3444, 3444), (3460, 3469), (3470, 3479)],
            "Mach": [(3510, 3519), (3520, 3529), (3530, 3530), (3531, 3531), (3532, 3532), (3533, 3533), (3534, 3534), (3535, 3535), (3536, 3536), (3538, 3538), (3540, 3549), (3550, 3559), (3560, 3569), (3580, 3580), (3581, 3581), (3582, 3582), (3585, 3585), (3586, 3586), (3589, 3589), (3590, 3599)],
            "ElcEq": [(3600, 3600), (3610, 3613), (3620, 3621), (3623, 3629), (3640, 3644), (3645, 3645), (3646, 3646), (3648, 3649), (3660, 3660), (3690, 3690), (3691, 3692), (3699, 3699)],
            "Autos": [(2296, 2296), (2396, 2396), (3010, 3011), (3537, 3537), (3647, 3647), (3694, 3694), (3700, 3700), (3710, 3710), (3711, 3711), (3713, 3713), (3714, 3714), (3715, 3715), (3716, 3716), (3792, 3792), (3790, 3791), (3799, 3799)],
            "Aero": [(3720, 3720), (3721, 3721), (3723, 3724), (3725, 3725), (3728, 3729)],
            "Ships": [(3730, 3731), (3740, 3743)],
            "Guns": [(3760, 3769), (3795, 3795), (3480, 3489)],
            "Gold": [(1040, 1049)],
            "Mines": [(1000, 1009), (1010, 1019), (1020, 1029), (1030, 1039), (1050, 1059), (1060, 1069), (1070, 1079), (1080, 1089), (1090, 1099), (1100, 1119), (1400, 1499)],
            "Coal": [(1200, 1299)],
            "Oil": [(1300, 1300), (1310, 1319), (1320, 1329), (1330, 1339), (1370, 1379), (1380, 1380), (1381, 1381), (1382, 1382), (1389, 1389), (2900, 2912), (2990, 2999)],
            "Util": [(4900, 4900), (4910, 4911), (4920, 4922), (4923, 4923), (4924, 4925), (4930, 4931), (4932, 4932), (4939, 4939), (4940, 4942)],
            "Telcm": [(4800, 4800), (4810, 4813), (4820, 4822), (4830, 4839), (4840, 4841), (4880, 4889), (4890, 4890), (4891, 4891), (4892, 4892), (4899, 4899)],
            "PerSv": [(7020, 7021), (7030, 7033), (7200, 7200), (7210, 7212), (7214, 7214), (7215, 7216), (7217, 7217), (7219, 7219), (7220, 7221), (7230, 7231), (7240, 7241), (7250, 7251), (7260, 7269), (7270, 7290), (7291, 7291), (7292, 7299), (7395, 7395), (7500, 7500), (7520, 7529), (7530, 7539), (7540, 7549), (7600, 7600), (7620, 7620), (7622, 7622), (7623, 7623), (7629, 7629), (7630, 7631), (7640, 7641), (7690, 7699), (8100, 8199), (8200, 8299), (8300, 8399), (8400, 8499), (8600, 8699), (8800, 8899), (7510, 7515)],
            "BusSv": [(2750, 2759), (3993, 3993), (7218, 7218), (7300, 7300), (7310, 7319), (7320, 7329), (7330, 7339), (7340, 7342), (7349, 7349), (7350, 7351), (7352, 7352), (7353, 7353), (7359, 7359), (7360, 7369), (7370, 7372), (7374, 7374), (7375, 7375), (7376, 7376), (7377, 7377), (7378, 7378), (7379, 7379), (7380, 7380), (7381, 7382), (7383, 7383), (7384, 7384), (7385, 7385), (7389, 7390), (7391, 7391), (7392, 7392), (7393, 7393), (7394, 7394), (7396, 7396), (7397, 7397), (7399, 7399), (7519, 7519), (8700, 8700), (8710, 8713), (8720, 8721), (8730, 8734), (8740, 8748), (8900, 8910), (8911, 8911), (8920, 8999), (4220, 4229)],
            "Comps": [(3570, 3579), (3680, 3680), (3681, 3681), (3682, 3682), (3683, 3683), (3684, 3684), (3685, 3685), (3686, 3686), (3687, 3687), (3688, 3688), (3689, 3689), (3695, 3695), (7373, 7373)],
            "Chips": [(3622, 3622), (3661, 3661), (3662, 3662), (3663, 3663), (3664, 3664), (3665, 3665), (3666, 3666), (3669, 3669), (3670, 3679), (3810, 3810), (3812, 3812)],
            "LabEq": [(3811, 3811), (3820, 3820), (3821, 3821), (3822, 3822), (3823, 3823), (3824, 3824), (3825, 3825), (3826, 3826), (3827, 3827), (3829, 3829), (3830, 3839)],
            "Paper": [(2520, 2549), (2600, 2639), (2670, 2699), (2760, 2761), (3950, 3955)],
            "Boxes": [(2440, 2449), (2640, 2659), (3220, 3221), (3410, 3412)],
            "Trans": [(4000, 4013), (4040, 4049), (4100, 4100), (4110, 4119), (4120, 4121), (4130, 4131), (4140, 4142), (4150, 4151), (4170, 4173), (4190, 4199), (4200, 4200), (4210, 4219), (4230, 4231), (4240, 4249), (4400, 4499), (4500, 4599), (4600, 4699), (4700, 4700), (4710, 4712), (4720, 4729), (4730, 4739), (4740, 4749), (4780, 4780), (4782, 4782), (4783, 4783), (4784, 4784), (4785, 4785), (4789, 4789)],
            "Whlsl": [(5000, 5000), (5010, 5015), (5020, 5023), (5030, 5039), (5040, 5042), (5043, 5043), (5044, 5044), (5045, 5045), (5046, 5046), (5047, 5047), (5048, 5048), (5049, 5049), (5050, 5059), (5060, 5060), (5063, 5063), (5064, 5064), (5065, 5065), (5070, 5078), (5080, 5080), (5081, 5081), (5082, 5082), (5083, 5083), (5084, 5084), (5085, 5085), (5086, 5087), (5088, 5088), (5090, 5090), (5091, 5092), (5093, 5093), (5094, 5094), (5099, 5099), (5100, 5100), (5110, 5113), (5120, 5122), (5130, 5139), (5140, 5149), (5150, 5159), (5160, 5169), (5170, 5172), (5180, 5182), (5190, 5199)],
            "Rtail": [(5200, 5200), (5210, 5219), (5220, 5229), (5230, 5231), (5250, 5251), (5260, 5261), (5270, 5271), (5300, 5300), (5310, 5311), (5320, 5320), (5330, 5331), (5334, 5334), (5340, 5349), (5390, 5399), (5400, 5400), (5410, 5411), (5412, 5412), (5420, 5429), (5430, 5439), (5440, 5449), (5450, 5459), (5460, 5469), (5490, 5499), (5500, 5500), (5510, 5529), (5530, 5539), (5540, 5549), (5550, 5559), (5560, 5569), (5570, 5579), (5590, 5599), (5600, 5699), (5700, 5700), (5710, 5719), (5720, 5722), (5730, 5733), (5734, 5734), (5735, 5735), (5736, 5736), (5750, 5799), (5900, 5900), (5910, 5912), (5920, 5929), (5930, 5932), (5940, 5940), (5941, 5941), (5942, 5942), (5943, 5943), (5944, 5944), (5945, 5945), (5946, 5946), (5947, 5947), (5948, 5948), (5949, 5949), (5950, 5959), (5960, 5969), (5970, 5979), (5980, 5989), (5990, 5990), (5992, 5992), (5993, 5993), (5994, 5994), (5995, 5995), (5999, 5999)],
            "Meals": [(5800, 5819), (5820, 5829), (5890, 5899), (7000, 7000), (7010, 7019), (7040, 7049), (7213, 7213)],
            "Banks": [(6000, 6000), (6010, 6019), (6020, 6020), (6021, 6021), (6022, 6022), (6023, 6024), (6025, 6025), (6026, 6026), (6027, 6027), (6028, 6029), (6030, 6036), (6040, 6059), (6060, 6062), (6080, 6082), (6090, 6099), (6100, 6100), (6110, 6111), (6112, 6113), (6120, 6129), (6130, 6139), (6140, 6149), (6150, 6159), (6160, 6169), (6170, 6179), (6190, 6199)],
            "Insur": [(6300, 6300), (6310, 6319), (6320, 6329), (6330, 6331), (6350, 6351), (6360, 6361), (6370, 6379), (6390, 6399), (6400, 6411)],
            "RlEst": [(6500, 6500), (6510, 6510), (6512, 6512), (6513, 6513), (6514, 6514), (6515, 6515), (6517, 6519), (6520, 6529), (6530, 6531), (6532, 6532), (6540, 6541), (6550, 6553), (6590, 6599), (6610, 6611)],
            "Fin": [(6200, 6299), (6700, 6700), (6710, 6719), (6720, 6722), (6723, 6723), (6724, 6724), (6725, 6725), (6726, 6726), (6730, 6733), (6740, 6779), (6790, 6791), (6792, 6792), (6793, 6793), (6794, 6794), (6795, 6795), (6798, 6798), (6799, 6799)],
            "Other": [(4950, 4959), (4960, 4961), (4970, 4971), (4990, 4991)]}
        for name, ranges in ff48_map.items():
            for low, high in ranges:
                if low <= s <= high:
                    return name
        return "Other"

    # def get_ff_sectors(self, sic):
    #     if pd.isna(sic) or sic <= 0:
    #         return "Other", "Other"
        
    #     s = int(sic)
    #     
        
    #     ff12 = self.get_ff12(s)
    #     ff48 = self.get_ff48(s)
    #     return ff12, ff48


    def build_stock_features(self, topk = 500) :
        """
        Perform backward price adjustment and calculate OHLCV for individual stocks.
        """
        target_df = self.load_universe_data()
        target_df['ret'] = pd.to_numeric(target_df['ret'], errors='coerce').fillna(0.0)

        target_df['mktcap_rank'] = target_df.groupby('date')['mktcap'].rank(method='first', ascending=False)
        target_df['is_tradable'] = (target_df['mktcap_rank'] <= topk).fillna(False).astype(int)
        target_df['siccd'] = pd.to_numeric(target_df['siccd'], errors='coerce').astype(int)
        # FF5 & FF12 & F48 sector classification based on SIC codes
        # https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/
        target_df['ff5_idx'] = pd.Categorical(target_df['siccd'].map(self.get_ff5)).codes
        target_df['ff12_idx'] = pd.Categorical(target_df['siccd'].map(self.get_ff12)).codes
        target_df['ff48_idx'] = pd.Categorical(target_df['siccd'].map(self.get_ff48)).codes
        print(target_df[['ff5_idx', 'ff12_idx', 'ff48_idx']].head())

        all_trading_dates = pd.to_datetime(np.sort(target_df['date'].unique()))

        def rebuild_backwards(g):
            g = g.set_index('date').reindex(all_trading_dates).reset_index().rename(columns={'index': 'date'})
    
            valid_price_indices = g.index[(g['prc'].notna()) & (g['prc'] != 0)]
            if valid_price_indices.empty:
                return None 
            first_idx = valid_price_indices[0]
            last_idx = valid_price_indices[-1]
            
            # Slice data for calculation
            valid_g = g.loc[first_idx:last_idx].copy()
            valid_g['ret'] = pd.to_numeric(valid_g['ret'], errors='coerce').fillna(0.0)
            
            # 4. Core backward adjustment logic
            base_price = np.abs(valid_g.loc[last_idx, 'prc'])
            growth = (1 + valid_g['ret']).cumprod()
            
            # Ratio of current growth to the end of the period growth for backward calculation
            valid_g['close'] = base_price * (growth / growth.iloc[-1])
            
            # Apply adjustment to OHLC and volume
            ratio = valid_g['close'] / valid_g['prc'].abs().replace(0, np.nan)
            valid_g['open'] = valid_g['openprc'].abs() * ratio
            valid_g['high'] = valid_g['askhi'].abs() * ratio
            valid_g['low'] = valid_g['bidlo'].abs() * ratio
            valid_g['volume'] = valid_g['vol'].abs() * valid_g['cfacshr']
            

            valid_g['close'] = valid_g['close'].ffill()
            valid_g['open'] = valid_g['open'].fillna(valid_g['close'])
            valid_g['high'] = valid_g['high'].fillna(valid_g['close'])
            valid_g['low'] = valid_g['low'].fillna(valid_g['close'])
            valid_g['volume'] = valid_g['volume'].fillna(0)

            
            res_cols = ['close', 'open', 'high', 'low', 'volume']
            g.loc[valid_g.index, res_cols] = valid_g[res_cols]
            
            g['is_tradable'] = g['is_tradable'].fillna(0)
            g['ff5_idx'] = g['ff5_idx'].ffill()
            g['ff12_idx'] = g['ff12_idx'].ffill()
            g['ff48_idx'] = g['ff48_idx'].ffill()


            return g
        final_full_data = target_df.groupby('permno').apply(rebuild_backwards).reset_index(drop=True)
        # Final Save
        file_path = os.path.join(self.output_dir, "final_master_data_adj.parquet")
        final_full_data.to_parquet(file_path, compression='snappy', index=False)

        del final_full_data
        del target_df
        gc.collect()
    
        

    def calculate_all_indicators(self, data: pd.DataFrame, predict_len=20) -> pd.DataFrame:
        """
        Calculate technical indicators using TA-Lib.
        Prices are normalized to ensure stationarity as preferred in academic literature.
        """
        data = data.dropna(subset=['close','high','low','volume'])
        close = data['close'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        volume = data['volume'].values.astype(float) + 1e-8 # Division by zero

        # 1. Price-based features: Log-Return and Log-Ratios
        data['return'] = np.log(data['close'] / data['close'].shift(predict_len))
        data['open_r'] = np.log(data['open'] / data['close'])
        data['high_r'] = np.log(data['high'] / data['close'])
        data['low_r'] = np.log(data['low'] / data['close'])
        
        # 2. Moving Averages and Trend (Price-Relative)
        # Calculate log-ratio against current price for stationarity
        data['SMA'] = np.log(talib.SMA(close, timeperiod=20) / close)
        data['EMA'] = np.log(talib.EMA(close, timeperiod=20) / close)
        data['WMA'] = np.log(talib.WMA(close, timeperiod=20) / close)
        data['SAR'] = np.log(talib.SAR(high, low) / close)
        
        # ADX is bounded (0-100), used as-is
        data['ADX'] = talib.ADX(high, low, close, timeperiod=14) 

        # 3. Momentum Indicators (Bounded Oscillators)
        data['RSI'] = talib.RSI(close, timeperiod=14)
        data['WilliamsR'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # MACD: Ratio-based normalization to prevent scale drift
        macd, signal, hist = talib.MACD(close)
        data['MACD'] = macd / close
        data['MACD_Signal'] = signal / close
        data['MACD_Hist'] = hist / close

        # 4. Volatility Indicators
        # Bollinger Bands: Calculate relative position within the band
        upper, middle, lower = talib.BBANDS(close, timeperiod=20)
        data['BB_Percent'] = (close - lower) / (upper - lower + 1e-8)
        data['BB_Upper'] = np.log(upper / close)
        data['BB_Middle'] = np.log(middle / close)
        data['BB_Lower'] = np.log(lower / close)
        
        data['ATR'] = talib.ATR(high, low, close, timeperiod=14) / close 

        # 5. Volume Indicators (Normalized)
        data['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        obv_values = talib.OBV(close, volume)
        obv_s = pd.Series(obv_values, index=data.index)
        
        # Normalize volume-based indicators by 20-day average volume
        vma20 = data['volume'].rolling(window=20).mean() + 1e-8
        obv_rolling_mu = obv_s.rolling(window=20).mean()
        data['OBV'] = (obv_s - obv_rolling_mu) / vma20
        data['ADOSC'] = talib.ADOSC(high, low, close, volume) / vma20
        
        # Stochastics
        slowk, slowd = talib.STOCH(high, low, close)
        data['Stoch_K'], data['Stoch_D'] = slowk , slowd
        data['ROC'] = talib.ROC(close, timeperiod=10)
        data['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26)
        data['CCI'] = talib.CCI(high, low, close, timeperiod=14) 
        
        data['MOM'] = talib.MOM(close, timeperiod=10) / close
        
        # Target Labeling with Delisting Return consideration
        data['close_delist'] = data['close'].copy()
        data['close_delist'] = data['close_delist'] * (1 + data['dlret'].fillna(0))
        future_val = data['close_delist'].shift(-predict_len)
        has_delisting = (data['dlret'] != 0).any()
        if has_delisting:
            data['future_close'] = future_val.fillna(data['close_delist'].iloc[-1])
        else:
            data['future_close'] = future_val
        data['LABEL'] = (data['future_close'] / data['close_delist'].shift(-1)) - 1

        # Testset label prediction length cutoff
        data = data.loc[pd.to_datetime(data.loc[:, 'date']) < self.cutoff_date]
        return data[self.warmup_time :] # MACD need 33 index // LABEL need 20 index


    def save_final(self, dataset):
        is_nan_row = dataset[self.cols + self.target + self.rel_cols].isna().any(axis=1)
        dataset.loc[is_nan_row, 'is_tradable'] = 0
        
        file_path = os.path.join(self.output_dir, "stock_dataset_final.parquet")
        dataset[self.save_cols].to_parquet(file_path, compression='snappy')
        valid_counts = dataset.loc[dataset["is_tradable"] == 1].groupby('date')['permno'].count()
        file_path = os.path.join(self.output_dir, "stock_list.csv")
        dataset[['ticker', 'siccd', 'primaryexch', 'permno', 'date']].ffill().bfill().sort_values('date').reset_index(drop=True).to_csv(file_path)

        print(f"Average valid stocks per day: {valid_counts.mean():.2f}")



if __name__ == "__main__":
    configs = parser.parse_args()
    topk = configs.topk
    pipeline = CRSPDATASET(topk=topk, start_year=2001, end_year=2025)
    
    pipeline.get_universe()
    pipeline.download_data_yearly()
    
    market_df = pipeline.build_market_features()
    
    pipeline.build_stock_features(topk = topk)
    
    master_data = pd.read_parquet(os.path.join(pipeline.output_dir, "final_master_data_adj.parquet"))
    
    print("Calculating technical indicators...")
    target_len = 20
    pipeline.cutoff_date = sorted(master_data['date'].unique())[-target_len]
    print(f"CUTOFF date : {pipeline.cutoff_date}")
    processed_data = master_data.groupby('permno').apply(pipeline.calculate_all_indicators,
                                                  predict_len = target_len).reset_index(drop=True)

    pipeline.save_final(processed_data)
    print("Full process complete.")