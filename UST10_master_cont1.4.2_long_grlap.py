import sys
sys.dont_write_bytecode = True
#%load_ext autoreload
#%autoreload 2
from utils_dir import get_curr_dir, include_home_dir
include_home_dir()

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader.data as web
import datetime
from datetime import timedelta

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

#from jumpmodels.jump_try_sp_clean import JumpModel                 # class of JM & CJM
from jumpmodels.jump_kernel_tr_grlap_gpu import JumpModel
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM

import warnings
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import time

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


hls = [5,63]
hlte = "5,63_rq+linear_ker+sp"

n_components = 2
jump_penalty = 1000
## Define JumpModel
jm_setting = dict(
    n_components=n_components,
    #jump_penalty=jump_penalty,#set in each application
    cont=True,
    #kernel = 'linear+rq',kernel_params = {'length_scale' : 100, 'alpha' : 20},#n_kernel_features = 1000, # set in each application
    use_spectral_penalty = True, sp_kernel = 'rq', sp_kernel_params = {'alpha': 20, 'length_scale':100}, normalize_laplacian = True,
    #use_weighted_jump_penalty = True, wjp_kernel = 'rq', wjp_kernel_params = {'alpha': 20, 'length_scale':100}, wjp_normalize_laplacian = True,
    )

df = pd.read_csv(datadict + assetfile + '.csv',index_col = 0)
df = df.reset_index()
df = df.rename(columns={"Date": "date"})
df["date"] = pd.to_datetime(df["date"])
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

start_date = datastart
end_date   = dataend

#rf_yield = web.DataReader(rfticker, "fred", start_date, end_date)
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
df["ret"] = np.log(df["TOT_RETURN_INDEX_GROSS_DVDS"]).diff() # log returns
data1 = df[["date", "ret"]].dropna()
bench = pd.Series(
    data1["ret"].values,
    index=data1["date"],
    name=benchticker
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
bi["bi5_logdiff_ewma"] = np.log(bi5_raw["PX_LAST"]).diff().ewm(halflife=5).mean()
bi["bi10_logdiff_ewma"] = np.log(bi10_raw["PX_LAST"]).diff().ewm(halflife=5).mean()
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

ext_features_pre = pd.concat([ext_features_pre, bi_aligned, 
                       move_aligned, 
                       vol_aligned], axis=1, join="inner")
ext_features_pre2 = ext_features_pre.drop(columns=["slope_diff_ewm"])
ext_features_pre_with_ns = pd.concat([ext_features_pre2, ns_aligned], axis=1, join="inner")
ext_features_pre_with_pc = pd.concat([ext_features_pre2, pc_aligned], axis=1, join="inner")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from xgboost import XGBRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits = 4)

#lambda_list = np.logspace(0, 1 ,4)
lambda_list = [0.5]

jm_setting = dict(
    n_components=n_components,
    #jump_penalty=jump_penalty,#set in each application
    cont=True,
    #kernel = 'linear+rq',kernel_params = {'length_scale' : 100, 'alpha' : 20},#n_kernel_features = 1000, # set in each application
    #use_spectral_penalty = True, sp_kernel = 'rq', sp_kernel_params = {'alpha': 20, 'length_scale':100}, normalize_laplacian = True,
    use_weighted_jump_penalty = True, wjp_kernel = 'rq', wjp_kernel_params = {'alpha': 20, 'length_scale':100}, wjp_normalize_laplacian = False,
    )

train_years = 11
pred_months = 6
#val_start = pd.Timestamp(valstart)
#val_end = pd.Timestamp(dataend)
val_start = pd.Timestamp('2014-01-01')
val_end = pd.Timestamp('2016-12-31')

clf_jmo_roll = []
results_jmo_roll = {}
perf_jmo_dict_roll = {}
regime_all = []

global_start = processed_train['base'].index.min()

patterns = {
    'base+ext_features'       : ('base', ext_features_pre),
    'NS+ext_features2'        : ('NS',   ext_features_pre2),
    'PCA+ext_features2'       : ('PCA',  ext_features_pre2),
    #'base+ext_features+ns'    : ('base', ext_features_pre_with_ns),
    #'base+ext_features+pc'    : ('base', ext_features_pre_with_pc),
}

for pname, (umodel, ext_features_df) in patterns.items():
    period_start = time.perf_counter()
    X_all      = pd.concat([processed_train[umodel],processed_val[umodel],processed_test[umodel]],axis = 0, ignore_index = False)
    X_all["time_idx"] = (X_all.index - global_start).days.astype(float)
    scaler = StandardScalerPD()
    X_all = scaler.fit_transform(X_all)
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
        best_alpha = None
        best_model = None

        for lam in lambda_list:
    
            cv_scores = []
            for train_idx, val_idx in tscv.split(Xj_train):
                X_tr_fold = Xj_train.iloc[train_idx]
                y_tr_fold = ret_all.loc[X_tr_fold.index]
                X_val_fold = Xj_train.iloc[val_idx]
                y_val_fold = ret_all.loc[X_val_fold.index]

                jm = JumpModel(**jm_setting, jump_penalty=lam, n_kernel_features = 1000)
                jm.fit(X_tr_fold, y_tr_fold, sort_by = "cumret")
                proba_jm = jm.proba_
                y_train = pd.Series(proba_jm.iloc[1:,1].values, index = X_tr_fold.index[:-1])

                idx_sh = X_val_fold.index[1:]
                p_tr = jm.predict_proba_online(X_val_fold.values)[:, 1]
                smooth = pd.Series(p_tr, index = X_val_fold.index,)#.ewm(halflife = 4, adjust = False).mean()
                y_pred = (smooth>0.5).astype(int)
                y_pred = pd.Series(y_pred[:-1].values, index = idx_sh)
                regime = 1 - y_pred

                ret_tr = ret_all.reindex(idx_sh)
                rf_tr = rf_all.reindex(idx_sh)
                strat = ret_tr.where(regime ==1, other = rf_tr)
                excess = strat - rf_tr
                mu = excess.mean() * 252
                sd = strat.std() * np.sqrt(252)
                sharpe = mu /sd if sd>0 else -np.inf
                cv_scores.append(mu)

            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_lambda = lam
                best_model = jm

        print(f"Prediction period: {pred_start} to {pred_end}")
        print(f"Best jump penalty: {best_lambda}, Best Score: {best_score:.3f}")


        Xj_pred = X_all.loc[pred_start:pred_end]
        pred_idx  = Xj_pred.index[1:]
        p_jmo = best_model.predict_proba_online(Xj_pred.values)
        smooth = pd.Series(p_jmo[:,1], index=Xj_pred.index)#.ewm(halflife=4, adjust=False).mean()
        y_pred = (smooth > 0.5).astype(int)
        y_pred = pd.Series(y_pred[:-1].values, index=pred_idx)
        regime     = 1 - y_pred
        regime_all = pd.concat([regime_all, regime])

        n_samples = Xj_pred.shape[0]
        n_kernel_features = min(500, n_samples // 2)
        jm_final = JumpModel(**jm_setting, jump_penalty = best_lambda, n_kernel_features = n_kernel_features)
        jm_final.fit(Xj_pred, ret_all, sort_by = "cumret")

        jm_true = pd.Series(jm_final.labels_, index = Xj_pred.index)
        #jm_true.loc[pred_idx].to_csv("jm_true.csv")
        #y_pred.to_csv("y_pred.csv")
        clf_report = classification_report(jm_true.loc[pred_idx],y_pred,target_names=["growth","crash"],labels = [0,1],zero_division=0)
        print("Classification Report:\n", clf_report)
    
        clf_dict = classification_report(jm_true.loc[pred_idx],y_pred.loc[jm_true[1:].index],target_names=["growth","crash"],labels = [0,1],zero_division=0, output_dict = True)
        n_samples = len(jm_true.loc[pred_idx])
        for label, metrics in clf_dict.items():
            if isinstance(metrics, dict):
                precision = metrics.get('precision',  0)
                recall    = metrics.get('recall',     0)
                f1_score  = metrics.get('f1-score',   0)
                support   = metrics.get('support',    0)
            else:
                precision = recall = f1_score = metrics
                support   = n_samples
            clf_jmo_roll.append({'pattern': pname,'label': label,'precision': precision,'recall': recall,'f1_score': f1_score,'support': support,})
        
        t0 += relativedelta(months = pred_months)
    
    pred_idx = regime_all.index    
    ret_val   = ret_ser.reindex(pred_idx).dropna()
    rf_aligned = rf_test.reindex(pred_idx).fillna(0.0)
    ret_strat  = ret_val.where(regime_all==1, other=rf_aligned)
    perf_jmo = compute_performance(
        ret_series=ret_strat,
        rf_series =rf_aligned,
        freq=252,
        benchmark=bench.reindex(pred_idx).fillna(0.0)
    )
    perf_jmo_dict_roll[name] = perf_jmo
    print(perf_jmo)

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

    cum_bh    = ret_val.cumsum()
    cum_jmo= ret_strat.cumsum()
    cum_rf    = rf_aligned.cumsum()
    cum_bench = bench.reindex(pred_idx).fillna(0.0).cumsum()
    period_elapsed = time.perf_counter() - period_start
    print(f"[Full loop] {pred_start.date()}～{pred_end.date()} total {period_elapsed:.1f}s")

    
    results_jmo_roll[pname] = {
        'best_params':         best_lambda,
        'best_cv_score':       best_score,
        'classification_report': clf_report,
        'perf_bh':             perf_bh,
        'perf_jmo':         perf_jmo,
        'cum_bh':              cum_bh,
        'cum_strat':           cum_jmo,
        'cum_rf' : cum_rf,
        'regime' : regime_all,
        'cum_bench' : cum_bench
    }

    fig, ax = plt.subplots()
    cum_bh.plot(label="Buy and Hold",ax = ax)
    cum_jmo.plot(label="JM Online shift Strategy",ax = ax)
    cum_rf.plot(label="Risk-Free Rate", linestyle="--", color="gray",ax = ax)
    cum_bench.plot(label="Benchmark", linestyle="--", color="red",ax = ax)

    ymin = min(cum_bh.min(), cum_jmo.min(), cum_rf.min(), cum_bench.min())
    ymax = max(cum_bh.max(), cum_jmo.max(), cum_rf.max(), cum_bench.max()) * 1.1

    ax.fill_between(
        cum_bh.index, ymin, ymax,
        where=(regime_all==0), alpha=0.3, color="red",   label="Bear Regime"
    )
    ax.fill_between(
        cum_bh.index, ymin, ymax,
        where=(regime_all==1), alpha=0.3, color="green", label="Bull Regime"
    )

    ax.legend(loc="upper left")
    ax.set_title(f"{pname}: Buy and Hold vs JM online shift")
    #plt.tight_layout()
    plt.show()
    plt.close()


import pickle
resdir = 'results'
nbname = 'UST10_master_cont1.4.2_long_grlap'

results_to_save = {}
for pname, res in results_jmo_roll.items():
    r = res.copy()
    #r.pop('figure', None)   # 図は含めない
    results_to_save[pname] = r

with open(f"{resdir}/{nbname}_results_jmo_roll.pkl", 'wb') as f:
    pickle.dump(results_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)


perf_table = []
for name, res in results_jmo_roll.items():
    row = {
        'pred':        "JMonline_roll",
        'model':       name,
        'annual_ret':res['perf_jmo']['annual_return'],
        'annual_vol':res['perf_jmo']['annual_vol'],
        'sharpe':    res['perf_jmo']['sharpe'],
        'max_drawdown': res['perf_jmo']['max_drawdown'],
        'sortino':   res['perf_jmo']['sortino'],
        'cvar_99':   res['perf_jmo']['cvar_99'],
        'calmar':    res['perf_jmo']['calmar'],
        'information_ratio': res['perf_jmo']['information_ratio'],
    }
    perf_table.append(row)
perf_df = pd.DataFrame(perf_table).set_index('model')
perf_df = perf_df.map(lambda cell: cell.item() if isinstance(cell, pd.Series) and cell.size == 1 else cell)
#display(perf_df)

plt.figure()
for name, res in results_jmo_roll.items():
    res['cum_strat'].plot(label=f"{name}-JM online shift", linewidth=2)
rf_test[pred_idx].cumsum().plot(label="Risk-Free Rate", linestyle='--', color='gray')
cum_bh.plot(label="Buy and Hold", linestyle=':', color='blue')
first = next(iter(results_jmo_roll.values()))
bench.reindex(first['cum_bh'].index).fillna(0).cumsum().plot(label="Benchmark", linestyle=':', color='red')

plt.legend()
plt.title("Cumulative Returns Comparison")
plt.show()

