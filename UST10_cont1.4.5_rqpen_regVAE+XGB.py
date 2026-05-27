import sys
sys.dont_write_bytecode = True
from utils_dir import get_curr_dir, include_home_dir
include_home_dir()

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta
import torch
import time
from dateutil.relativedelta import relativedelta
from sklearn.metrics import classification_report


import matplotlib as mpl
mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
})
mpl.use("Agg")
import matplotlib.pyplot as plt

from jumpmodels.utils import filter_date_range        # useful helpers
from feature import DataLoader
from jumpmodels.preprocess import DataClipperStd, StandardScalerPD
from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

from jumpmodels.jump_kernel_spect_dpen_v1 import JumpModelGPU as JumpModel  # class of Jump Model
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

datadict = 'data/'
asset = 'UST10'
assetfile = 'TRXVUSGOV10U(TOT_RETURN)_import'
datastart = '1982-01-11'
valstart = '1994-01-01'
teststart = '2004-01-01'
dataend = '2025-03-31'

pcafile = "pca_scores_dynamic_ewrb5.csv"
nsfile = "LSC_output_USD_1980_2025.csv"

rfticker  = "DGS3MO"
benchfile = 'data/lbustruu.csv'
benchticker = "AggBond"
benchfile2 = 'data/sptr.csv'
benchticker2 = "SP500"

# extention for multiple setting analysis
pca5 = "_pca5"
ns = "ns"
pca5online = "_pca5online"
nsonline = "nsonline"

ext_features_file = "data/ext_features_pre.csv"
bi5_file = "data/usggbe05.csv"
bi10_file = "data/usggbe10.csv"
move_file = "data/move.csv"
volfile = "UST10GARCH_vol.csv"
volfile2 = 'UST10_Hist21_vol.csv'
spvolfile = "data/SP500_Hist21_vol.csv"
spvolfile2 = 'data/SP500_GARCH_vol.csv'
corrfile = "data/SP500UST10_corr.csv"
sw3m_file = "data/usosfr3m.csv"
sw2_file = "data/usosfr02.csv"
sw10_file = "data/usosfr10.csv"

hls = [5,63]
hlte = "5,63_rqsp_onlyprobspectral_tscv"

tr_cost = 0.0005
n_components = 2
jump_penalty = 1000
## Define JumpModel
jm_setting = dict(
    n_components=n_components,
    #jump_penalty=jump_penalty,#set in each application
    cont=True,
    #kernel = 'linear+rq',kernel_params = {'length_scale' : 100, 'alpha' : 20},#n_kernel_features = 1000, # set in each application
    #use_spectral_penalty = True, sp_kernel = 'rq', sp_kernel_params = {'alpha': 20, 'length_scale':5}, normalize_laplacian = True,
    #use_weighted_jump_penalty = True, wjp_kernel = 'rq', wjp_kernel_params = {'alpha': 20, 'length_scale':100}, wjp_normalize_laplacian = True,
    )
## Data Preparation
df = pd.read_csv(datadict + assetfile + '.csv',index_col = 0)
df = df.reset_index()
df = df.rename(columns={"Date": "date"})
df['date'] = pd.to_datetime(df['date'])#, format='%d/%m/%Y')
df = df.sort_values("date", ascending=True)
df["ret"] = np.log(df[".TRXVUSGOV10U (TOT_RETURN)"]).diff() # log returns
data1 = df[["date", "ret"]].dropna()
ret_ser = pd.Series(
    data1["ret"].values,
    index=data1["date"],
    name=asset
)

class DataObj:
    pass

data_obj = DataObj()
data_obj.ret = ret_ser
pd.to_pickle(data_obj, datadict + asset + ".pkl")

class DataObj:
    pass
# daily data is from 1980-01-01
dl = DataLoader(ticker=asset, ver="v0").load(start_date=datastart, end_date=dataend)

print("Daily returns stored in `data.ret_ser`:", "-"*50, sep="\n")
print(dl.ret_ser, "-"*50, sep="\n")
print("Features stored in `data.X`:", "-"*50, sep="\n")
print(dl.X)

# add nsloading features
pcload_raw = pd.read_csv(pcafile, index_col=0, parse_dates=True)
start, end = datastart, dataend
PC1 = pcload_raw.loc[start:end, "PC1"]
PC2 = pcload_raw.loc[start:end, "PC2"]
PC3  = pcload_raw.loc[start:end, "PC3"]
PC4 = pcload_raw.loc[start:end, "PC4"]
PC5 = pcload_raw.loc[start:end, "PC5"]

pcload = pd.concat([PC1, PC2, PC3,PC4, PC5], axis=1)
pcfeatures = pd.DataFrame(index = pcload.index)

for hl in hls:
    #pcfeatures[f"PC1_ewma_{hl}"] = pcload["PC1"].ewm(halflife=hl).mean()
    pcfeatures[f"PC1_diff_ewma_{hl}"] = pcload["PC1"].diff().ewm(halflife=hl).mean()
    #pcfeatures[f"PC2_ewma_{hl}"] = pcload["PC2"].ewm(halflife=hl).mean()
    pcfeatures[f"PC2_diff_ewma_{hl}"] = pcload["PC2"].diff().ewm(halflife=hl).mean()
    #pcfeatures[f"PC3_ewma_{hl}"] = pcload["PC3"].ewm(halflife=hl).mean()
    pcfeatures[f"PC3_diff_ewma_{hl}"] = pcload["PC3"].diff().ewm(halflife=hl).mean()
    #pcfeatures[f"PC4_ewma_{hl}"] = pcload["PC4"].ewm(halflife=hl).mean()
    pcfeatures[f"PC4_diff_ewma_{hl}"] = pcload["PC4"].diff().ewm(halflife=hl).mean()
    #pcfeatures[f"PC5_ewma_{hl}"] = pcload["PC5"].ewm(halflife=hl).mean()
    pcfeatures[f"PC5_diff_ewma_{hl}"] = pcload["PC5"].diff().ewm(halflife=hl).mean()
pcfeatures = pcfeatures.dropna()
pc_df = pcfeatures
#missing_dates = dl.X.index.difference(pc_df.index)
pc_aligned = pc_df.reindex(dl.X.index, method="ffill")
X_with_pc = pd.concat([dl.X, pc_aligned], axis=1, join = "inner")
X_with_pc = X_with_pc.fillna(method="bfill")
# add nsloading features
nsload_raw = pd.read_csv(nsfile, index_col=0, parse_dates=True)
start, end = datastart, dataend
level = nsload_raw.loc[start:end, "Level"]
slope = nsload_raw.loc[start:end, "Slope"]
curv  = nsload_raw.loc[start:end, "Curvature"]

nsload = pd.concat([level, slope, curv], axis=1)
nsfeatures = pd.DataFrame(index = nsload.index)

for hl in hls:
    #nsfeatures[f"level_ewma_{hl}"] = nsload["Level"].ewm(halflife=hl).mean()
    nsfeatures[f"level_diff_ewma_{hl}"] = nsload["Level"].diff().ewm(halflife=hl).mean()
    #nsfeatures[f"slope_ewma_{hl}"] = nsload["Slope"].ewm(halflife=hl).mean()
    nsfeatures[f"slope_diff_ewma_{hl}"] = nsload["Slope"].diff().ewm(halflife=hl).mean()
    #nsfeatures[f"curv_ewma_{hl}"] = nsload["Curvature"].ewm(halflife=hl).mean()
    nsfeatures[f"curv_diff_ewma_{hl}"] = nsload["Curvature"].diff().ewm(halflife=hl).mean()
#nsfeatures["level"] = nsload["Level"]
#nsfeatures["level_diff"] = nsload["Level"].diff().fillna(0)
#nsfeatures["slope"] = nsload["Slope"]
#nsfeatures["slope_diff"] = nsload["Slope"].diff().fillna(0)
#nsfeatures["curv"] = nsload["Curvature"]
#nsfeatures["curv_diff"] = nsload["Curvature"].diff().fillna(0)
nsfeatures = nsfeatures.dropna()

ns_df = nsfeatures
#missing_dates = dl.X.index.difference(ns_df.index)
ns_aligned = ns_df.reindex(dl.X.index, method="ffill")
X_with_ns = pd.concat([dl.X, ns_aligned], axis=1, join = "inner")
X_with_ns = X_with_ns.fillna(method="bfill")
## Train Test Split
settings = {
    'base'      : dl.X,
    'NS'        : X_with_ns,
    'PCA'       : X_with_pc,
}
train_splits = {}
val_splits   = {}
test_splits  = {}
clipped_train = {}
clipped_val    = {}
clipped_test  = {}
processed_train = {}
processed_val    = {}
processed_test  = {}
inf_nan_report = {}

datastart  = pd.to_datetime(datastart)
valstart   = pd.to_datetime(valstart)
teststart  = pd.to_datetime(teststart)
dataend    = pd.to_datetime(dataend)


for name, X in settings.items():

    X_train = filter_date_range(X, start_date=datastart, end_date=valstart- timedelta(days=1))
    X_val  = filter_date_range(X, start_date=valstart, end_date=teststart- timedelta(days=1))
    X_test = filter_date_range(X, start_date=teststart, end_date=dataend)

    train_splits[name] = X_train
    val_splits[name]   = X_val
    test_splits[name]  = X_test

    clipper = DataClipperStd(mul=4.)
    Xc_train = clipper.fit_transform(X_train)
    Xc_val   = clipper.transform(X_val)
    Xc_test  = clipper.transform(X_test)

    def clean_df(df):
        df2 = df.replace([np.inf, -np.inf], np.nan)
        return df2.fillna(df2.mean())
    Xc_train_clean = clean_df(Xc_train)
    Xc_val_clean   = clean_df(Xc_val)
    Xc_test_clean  = clean_df(Xc_test)

    scaler = StandardScalerPD()
    X_train_proc = scaler.fit_transform(Xc_train_clean)
    X_val_proc   = scaler.transform(Xc_val_clean)
    X_test_proc  = scaler.transform(Xc_test_clean)

    clipped_train[name]    = Xc_train
    clipped_val[name]      = Xc_val
    clipped_test[name]     = Xc_test
    processed_train[name]  = X_train_proc
    processed_val[name]    = X_val_proc
    processed_test[name]   = X_test_proc

    def report(df):
        arr = df.values if hasattr(df, "values") else df
        return {
            'has_inf': np.isinf(arr).any(),
            'has_nan': np.isnan(arr).any()
        }
    inf_nan_report[name] = {
        'train_raw': report(Xc_train),
        'train_clean': report(Xc_train_clean),
        'val_raw': report(Xc_val),
        'val_clean': report(Xc_val_clean),
        'test_raw': report(Xc_test),
        'test_clean': report(Xc_test_clean),
    }

    ts, te = X_train.index[[0, -1]]
    us, ue = X_val.index [[0, -1]]
    vs, ve = X_test.index [[0, -1]]
    rpt = inf_nan_report[name]
    print(f"[{name}] train: {ts}〜{te} | validation {us} ~ {ue} |test: {vs}〜{ve}")
    print(f"  raw train—inf? {rpt['train_raw']['has_inf']}, nan? {rpt['train_raw']['has_nan']}")
    print(f"  clean train—inf? {rpt['train_clean']['has_inf']}, nan? {rpt['train_clean']['has_nan']}")
    print(f"  raw val   —inf? {rpt['val_raw']['has_inf']}, nan? {rpt['val_raw']['has_nan']}")
    print(f"  clean val —inf? {rpt['val_clean']['has_inf']}, nan? {rpt['val_clean']['has_nan']}")
    print(f"  raw test —inf? {rpt['test_raw']['has_inf']}, nan? {rpt['test_raw']['has_nan']}")
    print(f"  clean test —inf? {rpt['test_clean']['has_inf']}, nan? {rpt['test_clean']['has_nan']}\n")
## Preparation for performance evaluation
start_date = datastart
end_date   = dataend

rf_yield = pd.read_pickle('data/rf_yield.pkl')
rf_yield = rf_yield.resample("B").ffill()
rf_yield = rf_yield / 100.0

rf_daily = np.log(1+rf_yield[rfticker]) / 252.0
rf_test = rf_daily
df = pd.read_csv(benchfile,index_col = 0, skiprows = 5)
df = df.reset_index()
df = df.rename(columns={"Date": "date"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date", ascending=True)
df["ret"] = np.log(df["PX_LAST"]).diff() # log returns
data1 = df[["date", "ret"]].dropna()
bench = pd.Series(
    data1["ret"].values,
    index=data1["date"],
    name=benchticker
)
df = pd.read_csv(benchfile2,index_col = 0, skiprows = 6)
df = df.reset_index()
df = df.rename(columns={"Date": "date"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date", ascending=True)
df["ret"] = np.log(df["PX_LAST"]).diff() # log returns
data2 = df[["date", "ret"]].dropna()
bench2 = pd.Series(
    data2["ret"].values,
    index=data2["date"],
    name=benchticker2
)
def compute_performance(
    ret_series: pd.Series,
    rf_series: pd.Series = None,
    freq: int = 252,
    benchmark: pd.Series = None
) -> pd.DataFrame:
    """
    """
    if rf_series is None:
        rf_aligned = pd.Series(0.0, index=ret_series.index)
    else:
        rf_aligned = rf_series.reindex(ret_series.index, method="ffill").fillna(0.0)

    excess_log_ret = ret_series - rf_aligned

    mu = excess_log_ret.mean()        # 日次超過対数リターンの平均
    sigma = ret_series.std(ddof=0)
    sharpe = np.nan
    if sigma > 0:
        sharpe = (mu * freq) / (sigma * np.sqrt(freq))

    ann_return = mu * freq
    ann_vol    = sigma * np.sqrt(freq)

    cumlog = excess_log_ret.cumsum()
    cumret = np.exp(cumlog)

    running_max = cumret.cummax()
    drawdown   = cumret / running_max - 1.0
    max_dd     = drawdown.min()

    down_ret = excess_log_ret.copy()
    down_ret[down_ret > 0] = 0.0
    dd_daily = down_ret.std(ddof=0)
    sortino = np.nan
    if dd_daily > 0:
        sortino = (mu * freq) / (dd_daily * np.sqrt(freq))

    var_99 = np.nanpercentile(excess_log_ret, 1)
    cvar_99 = excess_log_ret[excess_log_ret <= var_99].mean()

    ir = np.nan
    if benchmark is not None:
        bp = benchmark.reindex(ret_series.index, method="ffill").fillna(0.0)
        active = ret_series - bp
        mu_a = active.mean()       # 平均超過リターン（日次）
        sigma_a = active.std(ddof=0)
        if sigma_a > 0:
            ir = (mu_a * freq) / (sigma_a * np.sqrt(freq))

    calmar = np.nan
    if max_dd != 0:
        calmar = ann_return / abs(max_dd)

    df = pd.DataFrame( {
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "sortino": sortino,
        "cvar_99": cvar_99,
        "calmar": calmar,
        "information_ratio": ir,
        #"cumret_series": cumret,    # 完整な累積資産価値（Series）も返す
        #"drawdown_series": drawdown
    },index=[0] )

    return df
## Explanatory Variables
ext_features_pre = pd.read_csv(ext_features_file, index_col=0, parse_dates=True)
ext_features_pre = ext_features_pre.reindex(dl.X.index, method="ffill")
bi5_raw = pd.read_csv(bi5_file,index_col = 0, skiprows = 5)
bi5_raw = bi5_raw.reset_index()
bi5_raw = bi5_raw.rename(columns={"Date": "date"})
bi5_raw["date"] = pd.to_datetime(bi5_raw["date"])
bi5_raw = bi5_raw.sort_values("date", ascending=True)
bi5_raw.set_index("date", inplace=True)

bi10_raw = pd.read_csv(bi10_file,index_col = 0, skiprows = 5)
bi10_raw = bi10_raw.reset_index()
bi10_raw = bi10_raw.rename(columns={"Date": "date"})
bi10_raw["date"] = pd.to_datetime(bi10_raw["date"])
bi10_raw = bi10_raw.sort_values("date", ascending=True)
bi10_raw.set_index("date", inplace=True)

bi =pd.DataFrame(index=bi10_raw.index)
#bi["bi2_logdiff_ewma"] = np.log(bi2_raw["PX_LAST"]).diff().ewm(halflife=5).mean()
bi["bi5_diff_ewma"] = bi5_raw["PX_LAST"].diff().ewm(halflife=5).mean()
bi["bi10_diff_ewma"] = bi10_raw["PX_LAST"].diff().ewm(halflife=5).mean()
bi["bi_slope_ewma"] = (bi10_raw["PX_LAST"] - bi5_raw["PX_LAST"]).diff().ewm(halflife=5).mean()
bi_aligned = bi.reindex(ext_features_pre.index, method="ffill")
#bi_aligned.isna().sum()
bi_aligned = bi_aligned.fillna(method="bfill")
move_raw = pd.read_csv(move_file, index_col = 0, skiprows = 6)
move_raw = move_raw.reset_index()
move_raw = move_raw.rename(columns={"Date": "date"})
move_raw["date"] = pd.to_datetime(move_raw["date"])
move_raw = move_raw.sort_values("date", ascending=True)
move_raw.set_index("date", inplace=True)
move = pd.DataFrame(index=move_raw.index)
move["move_logdiff_ewma"] = np.log(move_raw["PX_LAST"]).diff().ewm(halflife=5).mean()
move_aligned = move.reindex(ext_features_pre.index, method="ffill")
vol_df = pd.read_csv(volfile, index_col=0)
vol_df = vol_df.reset_index()
vol_df = vol_df.sort_values("date", ascending=True)
vol_df["date"] = pd.to_datetime(vol_df["date"])
vol_df.set_index("date",inplace = True)
vol = pd.DataFrame(index = vol_df.index)
vol["vol_diff_ewma"] = vol_df["volatility"].diff().ewm(halflife=5).mean()
vol_aligned = vol.reindex(ext_features_pre.index, method="ffill")
spvol_df = pd.read_csv(spvolfile, index_col=0)
spvol_df = spvol_df.reset_index()
spvol_df = spvol_df.sort_values("date", ascending=True)
spvol_df["date"] = pd.to_datetime(spvol_df["date"])
spvol_df.set_index("date", inplace=True)
spvol = pd.DataFrame(index=spvol_df.index)
spvol["spvol_diff_ewma"] = spvol_df["hsvol"].diff().ewm(halflife=5).mean()
spvol_aligned = spvol.reindex(ext_features_pre.index, method="ffill")
corr_df =pd.read_csv(corrfile, index_col=0)
corr_df = corr_df.reset_index()
corr_df = corr_df.sort_values("date", ascending=True)
corr_df["date"] = pd.to_datetime(corr_df["date"])
corr_df.set_index("date", inplace=True)
corr = pd.DataFrame(index=corr_df.index)
corr["corr_diff_ewma"] = corr_df["corr"].diff().ewm(halflife=5).mean()
corr_aligned = corr.reindex(ext_features_pre.index, method="ffill")

sw02_raw = pd.read_csv(sw2_file,index_col = 0, skiprows = 7)
sw02_raw = sw02_raw.reset_index()
sw02_raw = sw02_raw.rename(columns={"Date": "date"})
sw02_raw["date"] = pd.to_datetime(sw02_raw["date"])
sw02_raw = sw02_raw.sort_values("date", ascending=True)
sw02_raw.set_index("date", inplace=True)

sw10_raw = pd.read_csv(sw10_file,index_col = 0, skiprows = 7)
sw10_raw = sw10_raw.reset_index()
sw10_raw = sw10_raw.rename(columns={"Date": "date"})
sw10_raw["date"] = pd.to_datetime(sw10_raw["date"])
sw10_raw = sw10_raw.sort_values("date", ascending=True)
sw10_raw.set_index("date", inplace=True)

sw3m_raw = pd.read_csv(sw3m_file,index_col = 0, skiprows = 7)
sw3m_raw = sw3m_raw.reset_index()
sw3m_raw = sw3m_raw.rename(columns={"Date": "date"})
sw3m_raw["date"] = pd.to_datetime(sw3m_raw["date"])
sw3m_raw = sw3m_raw.sort_values("date", ascending=True)
sw3m_raw.set_index("date", inplace=True)

sw =pd.DataFrame(index=sw10_raw.index)
sw["sw2_diff_ewma"] = sw02_raw["PX_LAST"].diff().ewm(halflife=5).mean()
sw["sw10_diff_ewma"] = sw10_raw["PX_LAST"].diff().ewm(halflife=5).mean()
sw["sw3m_diff_ewma"] = sw3m_raw["PX_LAST"].diff().ewm(halflife=5).mean()
sw["sw_slope_ewma"] = (sw10_raw["PX_LAST"] - sw02_raw["PX_LAST"]).diff().ewm(halflife=5).mean()
sw_aligned = sw.reindex(ext_features_pre.index, method="ffill")
#bi_aligned.isna().sum()
sw_aligned = sw_aligned.fillna(method="bfill")
ext_features_pre = pd.concat([ext_features_pre, bi_aligned, 
                       move_aligned, 
                       vol_aligned,
                       #spvol_aligned,
                       corr_aligned,
                       ], axis=1, join="inner")
ext_features_pre2 = ext_features_pre.drop(columns=["slope_diff_ewm"])
ext_features_pre_with_ns = pd.concat([ext_features_pre2, ns_aligned], axis=1, join="inner")
ext_features_pre_with_pc = pd.concat([ext_features_pre2, pc_aligned], axis=1, join="inner")

jm_setting = dict(
    n_components=n_components,
    #jump_penalty=jump_penalty,#set in each application
    cont=True,
    #kernel = 'linear+rq',kernel_params = {'alpha': 20, 'length_scale':5},#n_kernel_features = 1000, # set in each application
    use_spectral_penalty = True, sp_kernel = 'rq', sp_kernel_params = {'alpha': 20, 'length_scale':100}, normalize_laplacian = True,
    #use_weighted_jump_penalty = True, wjp_kernel = 'rq', wjp_kernel_params = {'alpha': 20, 'length_scale':100}, wjp_normalize_laplacian = True,
    )

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.dummy import DummyClassifier
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit

lambda_list = np.logspace(4.5, 5, 3)
n_components_list = [2, 6, 10, 14, 18]

param_grid = {'lam': lambda_list, 'n_comp': n_components_list}
grid = list(ParameterGrid(param_grid))

train_years = 11
pred_months = 6
val_start = pd.Timestamp(valstart)
val_end = pd.Timestamp(dataend)

clf_xgb_roll = []
results_xgb_roll = {}
perf_jm_xgb_dict_roll = {}
regime_all = []

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class SemiSupervisedVAE(nn.Module):
    def __init__(self, n_feat, z_dim, n_classes=2):
        super().__init__()
        # Encoder: x -> (μ, logσ)
        self.enc = nn.Sequential(
            nn.Linear(n_feat, 128), nn.ReLU(),
            nn.Linear(128, 2*z_dim)
        )
        # Decoder: z -> x̂
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, n_feat)
        )
        # Classification head: z -> y_pred_logits
        self.cls = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        # 1) エンコード
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=1)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        # 2) 再構成
        x̂ = self.dec(z)
        # 3) 分類予測
        logits = self.cls(z)
        return x̂, mu, logvar, logits



# クロスバリデーションを回して平均スコアを返す関数
def evaluate_params(lam, n_comp, Xj_train, ret_all, rf_all, ext_features_df, jm_setting, tr_cost):
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(Xj_train):
        # --- 各フォールドの処理をここに ---
        X_tr_fold, y_tr_fold = Xj_train.iloc[train_idx], ret_all.loc[Xj_train.index[train_idx]]
        X_val_fold, y_val_fold = Xj_train.iloc[val_idx], ret_all.loc[Xj_train.index[val_idx]]
        
        # JumpModel のフィッティング
        jm = JumpModel(**jm_setting, jump_penalty=lam, n_kernel_features=min(500, len(train_idx)//2))
        jm.fit(X_tr_fold, y_tr_fold, sort_by="cumret")
        label_jm = pd.Series(jm.labels_, index=X_tr_fold.index)
        y_train = pd.Series(label_jm.iloc[1:].values, index = X_tr_fold.index[:-1])
        
        if len(y_train.unique()) < 2:
            continue
        
        X_feat_tr = pd.concat([X_tr_fold.loc[X_tr_fold.index[:-1]], ext_features_df.reindex(X_tr_fold.index[:-1])], axis = 1)
        X_feat_tr = X_feat_tr.dropna(axis=1, how='any')
        x_tr_col = X_feat_tr.columns
        X_feat_val = pd.concat([X_val_fold, ext_features_df.reindex(X_val_fold.index)], axis=1)
        X_feat_val = X_feat_val.dropna(axis=1, how='any')
        X_feat_val = X_feat_val.reindex(columns=x_tr_col)
        
        y_train_bin = (y_train > 0.5).astype(int)

        scaler = StandardScaler()
        X_tr_mat   = scaler.fit_transform(X_feat_tr.values)   # (n_samples, n_feats)
        y_tr_bin   = y_train_bin.values                       # (n_samples,)
        X_val_mat = scaler.transform(X_feat_val.values)       # (m_samples, n_feats)

        ds_tr = TensorDataset(
            torch.from_numpy(X_tr_mat).float(),
            torch.from_numpy(y_tr_bin).long()
        )
        loader = DataLoader(ds_tr, batch_size=128, shuffle=True)
        model  = SemiSupervisedVAE(n_feat=X_tr_mat.shape[1], z_dim=n_comp, n_classes=2).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 学習ループ (ELBO + CE)
        recon_loss = nn.MSELoss(reduction='sum')
        ce_loss    = nn.CrossEntropyLoss()
        for epoch in range(80):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                x̂, mu, logvar, logits = model(xb)

                # ELBO
                L_recon = recon_loss(x̂, xb)
                L_kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # 分類損失
                L_ce    = ce_loss(logits, yb)

                u = mu[1:] - mu[:-1]               # (B-1, z_dim)
                tv_l1 = u.abs().mean()             # L1（TV）
                # tv_l2 = u.pow(2).mean()
                p = torch.softmax(logits, dim=1)[:, 1]     # (B,) クラス1の確率
                p_t, p_tm1 = p[1:], p[:-1]
                eps = 1e-6
                p_t  = p_t.clamp(eps, 1-eps); p_tm1 = p_tm1.clamp(eps, 1-eps)
                kl = p_t * (p_t/p_tm1).log() + (1-p_t)*((1-p_t)/(1-p_tm1)).log()

                #loss = L_recon + L_kl + 10.0 * L_ce +10 * tv_l1
                loss = L_recon + L_kl + 50 * L_ce + 50 * kl.mean() * 50

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            #print(f"Epoch {epoch} loss={total_loss/len(ds_tr):.3f}")

        model.eval()
        with torch.no_grad():
            X_tr_t = torch.from_numpy(X_tr_mat).float().to(device)
            _, mu_tr, _ , _ = model(X_tr_t)
            Z_tr = mu_tr.cpu().numpy()    # shape=(n_tr, z_dim)

            X_val_t = torch.from_numpy(X_val_mat).float().to(device)
            _, mu_val, _ , _ = model(X_val_t)
            Z_val = mu_val.cpu().numpy()  # shape=(n_val, z_dim)

        xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
        xgb.fit(Z_tr, y_tr_bin)
        y_val_pred = xgb.predict_proba(Z_val)[:,1]

        idx_sh = X_feat_val.index[1:]
        smooth = pd.Series(y_val_pred, index = X_feat_val.index).ewm(halflife = 4, adjust = False).mean()
        y_pred = (smooth>0.5).astype(int)
        y_pred = pd.Series(y_pred[:-1].values, index = idx_sh)
        regime = 1 - y_pred
        
        ret_tr = ret_ser.reindex(idx_sh)
        rf_tr = rf_all.reindex(idx_sh)
        #ret2_aligned = ret_ser2.reindex(idx_sh, method="ffill")
        strat = ret_tr.where(regime ==1, other = rf_tr)
        switch = (regime != regime.shift(1)).astype(float)  # True の日は 1.0, それ以外は 0.0
        cost  = switch * tr_cost
        strat = strat - cost

        excess = strat - rf_tr
        mu = excess.mean() * 252
        sd = strat.std() * np.sqrt(252)
        sharpe = mu /sd if sd>0 else -np.inf
        cv_scores.append(mu)
    return lam, n_comp, np.mean(cv_scores)


patterns = {
    #'base+ext_features'       : ('base', ext_features_pre),
    'NS+ext_features2'        : ('NS',   ext_features_pre2),
    #'PCA+ext_features2'       : ('PCA',  ext_features_pre2),
    'base+ext_features+ns'    : ('base', ext_features_pre_with_ns),
    'base+ext_features+pc'    : ('base', ext_features_pre_with_pc),
}

for pname, (umodel, ext_features_df) in patterns.items():
    period_start = time.perf_counter()
    X_all      = pd.concat([processed_train[umodel],processed_val[umodel],processed_test[umodel]],axis = 0, ignore_index = False)
    ret_all    = dl.ret_ser
    rf_all     = rf_test
    regime_all = pd.Series(dtype=int)
    jm_labels = pd.Series(dtype=int)

    t0 = val_start
    while t0 + relativedelta(months=pred_months) <= val_end:
        train_start = t0 - relativedelta(years=train_years)
        train_end   = t0 - timedelta(days=1)
        pred_start  = t0
        pred_end    = t0 + relativedelta(months=pred_months) - timedelta(days=1)

        Xj_train = X_all.loc[train_start:train_end]
        #idx_train = Xj_train.index[:-1]

        best_score = -np.inf
        best_lambda = None
        #best_alpha = None
        best_model = None
        best_n_comp = None
        
        # n_jobs=-1 で全CPUコアを使って並列実行
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_params)(
                params['lam'], params['n_comp'],
                Xj_train, ret_all, rf_all, ext_features_df, jm_setting, tr_cost
            )
            for params in grid
        )
        
        print(f"Prediction period: {pred_start} to {pred_end}")
        best_lambda, best_n_comp, best_score = max(results, key=lambda x: x[2])
        print(f"Best: jump penalty ={best_lambda}, n_comp ={best_n_comp}, score={best_score:.3f}")

        n_samples = Xj_train.shape[0]
        n_kernel_features = min(1000, n_samples // 2)
        jm_final = JumpModel(**jm_setting,jump_penalty=best_lambda,n_kernel_features=n_kernel_features)
        jm_final.fit(Xj_train, ret_all, sort_by="cumret")

        label_jm = jm_final.labels_
        y_train_full = pd.Series(label_jm.iloc[1:].values,index = Xj_train.index[:-1])
        y_tr_full_bin   = y_train_full.values
        X_feat_full = pd.concat([Xj_train.loc[Xj_train.index[:-1]],ext_features_df.reindex(Xj_train.index[:-1])], axis=1)
        X_feat_full = X_feat_full.dropna(axis=1, how='any')
        
        feature_cols = X_feat_full.columns
        
        scaler_full = StandardScaler().fit(X_feat_full.values)
        X_tr_mat_full = scaler_full.transform(X_feat_full.values)
        
        Xj_pred = X_all.loc[pred_start:pred_end]
        X_feat_val_full = pd.concat([
            Xj_pred,
            ext_features_df.reindex(Xj_pred.index)
        ], axis=1).reindex(columns=feature_cols)  # ← 列を揃える
        X_val_mat_full = scaler_full.transform(X_feat_val_full.values)
    
        ds_tr = TensorDataset(
            torch.from_numpy(X_tr_mat_full).float(),
            torch.from_numpy(y_tr_full_bin).long()
        )
        loader = DataLoader(ds_tr, batch_size=128, shuffle=True)
        model  = SemiSupervisedVAE(n_feat=X_tr_mat_full.shape[1], z_dim=best_n_comp, n_classes=2).to(device)
        opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    
        # 学習ループ (ELBO + CE)
        recon_loss = nn.MSELoss(reduction='sum')
        ce_loss    = nn.CrossEntropyLoss()
        for epoch in range(80):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                x̂, mu, logvar, logits = model(xb)
    
                # ELBO
                L_recon = recon_loss(x̂, xb)
                L_kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                # 分類損失
                L_ce    = ce_loss(logits, yb)
    
                u = mu[1:] - mu[:-1]               # (B-1, z_dim)
                tv_l1 = u.abs().mean()             # L1（TV）
                # tv_l2 = u.pow(2).mean()
                p = torch.softmax(logits, dim=1)[:, 1]     # (B,) クラス1の確率
                p_t, p_tm1 = p[1:], p[:-1]
                eps = 1e-6
                p_t  = p_t.clamp(eps, 1-eps); p_tm1 = p_tm1.clamp(eps, 1-eps)
                kl = p_t * (p_t/p_tm1).log() + (1-p_t)*((1-p_t)/(1-p_tm1)).log()
    
                #loss = L_recon + L_kl + 10.0 * L_ce +10 * tv_l1
                loss = L_recon + L_kl + 50 * L_ce + 50 * kl.mean() * 50
    
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            #print(f"Epoch {epoch} loss={total_loss/len(ds_tr):.3f}")
    
        model.eval()
        with torch.no_grad():
            X_tr_t = torch.from_numpy(X_tr_mat_full).float().to(device)
            _, mu_tr, _ , _ = model(X_tr_t)
            Z_tr = mu_tr.cpu().numpy()    # shape=(n_tr, z_dim)
    
            X_val_t = torch.from_numpy(X_val_mat_full).float().to(device)
            _, mu_val, _ , _ = model(X_val_t)
            Z_val = mu_val.cpu().numpy()  # shape=(n_val, z_dim)
    
        xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
        xgb.fit(Z_tr, y_tr_full_bin)
        y_val_pred_full = xgb.predict_proba(Z_val)[:,1]
    
        pred_idx = X_feat_val_full.index[1:]
        smoothed  = pd.Series(y_val_pred_full, index=X_feat_val_full.index).ewm(halflife=4, adjust=False).mean()
        y_pred    = (smoothed > 0.5).astype(int)
        y_pred    = pd.Series(y_pred[:-1].values, index=pred_idx)  # 翌日シフト
        regime     = 1 - y_pred
        regime_all = pd.concat([regime_all, regime])
    
        # 確保インデックスと予測
        jm_chk = JumpModel(**jm_setting, jump_penalty=best_lambda, n_kernel_features = 1000)
        jm_chk.fit(Xj_pred, dl.ret_ser, sort_by="cumret")
        labels_test = pd.Series(jm_chk.labels_, index=Xj_pred.index)
    
        clf_report = classification_report(labels_test.loc[pred_idx],y_pred.loc[labels_test[1:].index],target_names=["growth","crash"],labels = [0,1],zero_division=0)
        print("Classification Report:\n", clf_report)
        
        clf_dict = classification_report(labels_test.loc[pred_idx],y_pred.loc[labels_test[1:].index],target_names=["growth","crash"],labels = [0,1],zero_division=0, output_dict = True)
        n_samples = len(labels_test.loc[pred_idx])
        for label, metrics in clf_dict.items():
            if isinstance(metrics, dict):
                precision = metrics.get('precision',  0)
                recall    = metrics.get('recall',     0)
                f1_score  = metrics.get('f1-score',   0)
                support   = metrics.get('support',    0)
            else:
                precision = recall = f1_score = metrics
                support   = n_samples
            clf_xgb_roll.append({
                'pattern': pname,
                'label': label,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'support': support,
            })
        
        t0 += relativedelta(months = pred_months)
    
    pred_idx = regime_all.index    
    ret_val   = ret_ser.reindex(pred_idx).dropna()
    rf_aligned = rf_test.reindex(pred_idx).fillna(0.0)
    #ret2_aligned = ret_ser2.reindex(pred_idx, method="ffill").fillna(0.0)
    ret_strat  = ret_val.where(regime_all==1, other=rf_aligned)
    switch = (regime_all != regime_all.shift(1)).astype(int)
    cost  = switch * tr_cost
    ret_strat = ret_strat - cost
    
    perf_jm_xgb = compute_performance(
        ret_series=ret_strat,
        rf_series =rf_aligned,
        freq=252,
        benchmark=bench.reindex(pred_idx).fillna(0.0)
    )
    perf_jm_xgb_dict_roll[name] = perf_jm_xgb
    print(perf_jm_xgb)

    perf_bh = compute_performance(
        ret_series = ret_val,
        rf_series  = rf_aligned,
        freq       = 252,
        benchmark  = bench.reindex(pred_idx).fillna(0.0)
    )
    print(perf_bh)

    perf_bench = compute_performance(
        ret_series = bench.reindex(pred_idx).fillna(0.0),
        rf_series  = rf_aligned,
        freq       = 252,
        benchmark  = bench.reindex(pred_idx).fillna(0.0)
    )
    print(perf_bench)
    
    perf_bench2 = compute_performance(
        ret_series = bench2.reindex(pred_idx).fillna(0.0),
        rf_series  = rf_aligned,
        freq       = 252,
        benchmark  = bench.reindex(pred_idx).fillna(0.0)
    )
    print(perf_bench2)
    

    cum_bh1    = ret_val.cumsum()
    cum_jm_xgb= ret_strat.cumsum()
    cum_rf    = rf_aligned.cumsum()
    cum_bench = bench.reindex(pred_idx).fillna(0.0).cumsum()
    cum_bench2 = bench2.reindex(pred_idx).fillna(0.0).cumsum()

    period_elapsed = time.perf_counter() - period_start
    print(f"[Full loop] {pred_start.date()}～{pred_end.date()} total {period_elapsed:.1f}s")


    # ９．結果格納
    results_xgb_roll[pname] = {
        'best_params':         best_lambda,
        'best_cv_score':       best_score,
        'classification_report': clf_dict,
        'perf_bh':             perf_bh,
        'perf_jm_xgb':         perf_jm_xgb,
        'perf_bench':         perf_bench,
        'perf_bench2':         perf_bench2,
        'cum_bh1':              cum_bh1,
        'cum_strat':           cum_jm_xgb,
        'regime':             regime_all,
        'cum_rf':              cum_rf,
        'cum_bench':           cum_bench,
        'cum_bench2':          cum_bench2,
    }


perf_table = []
for name, res in results_xgb_roll.items():
    row = {
        'pred':        "XGB_roll",
        'model':       name,
        'annual_ret':res['perf_jm_xgb']['annual_return'],
        'annual_vol':res['perf_jm_xgb']['annual_vol'],
        'sharpe':    res['perf_jm_xgb']['sharpe'],
        'max_drawdown': res['perf_jm_xgb']['max_drawdown'],
        'sortino':   res['perf_jm_xgb']['sortino'],
        'cvar_99':   res['perf_jm_xgb']['cvar_99'],
        'calmar':    res['perf_jm_xgb']['calmar'],
        'information_ratio': res['perf_jm_xgb']['information_ratio'],
    }
    perf_table.append(row)
perf_df = pd.DataFrame(perf_table).set_index('model')
perf_df = perf_df.map(lambda cell: cell.item() if isinstance(cell, pd.Series) and cell.size == 1 else cell)


import pickle
resdir = 'results'
nbname = 'UST10_cont1.4.5_GPU_rqpen_regVAE+XGB'

results_to_save = {}
for pname, res in results_xgb_roll.items():
    r = res.copy()
    r.pop('figure', None)   # 図は含めない
    results_to_save[pname] = r

with open(f"{resdir}/{nbname}_results_xgb_roll.pkl", 'wb') as f:
    pickle.dump(results_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
print(rf_aligned.mean() * 252)
