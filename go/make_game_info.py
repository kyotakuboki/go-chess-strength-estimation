import argparse
import time
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, cpu_count, delayed

def _safe_n_jobs(user_jobs):
    if user_jobs and user_jobs > 0:
        return user_jobs
    return max(1, min(4, cpu_count() - 1))

def _downcast_numeric(df):
    float_cols = df.select_dtypes("float64").columns
    int_cols = df.select_dtypes("int64").columns
    df[float_cols] = df[float_cols].astype("float32")
    df[int_cols] = df[int_cols].astype("int32")
    return df

_ALPHA = 1e-10
_PRIORS = [f"prior_{i}" for i in ['9d','7d','5d','3d','1d','2k','4k','6k','8k','10k']] + ['prior_kt']

def compute_prior(df):
    cols = _PRIORS 
    data = df
    data["Target"] = np.where(data["turn"] % 2 == 0, data["PB"], data["PW"])

    for c in cols:
        data[f"log_{c}"] = np.log(data[c] + _ALPHA)
        
    grp_keys = ["id", "Target"]
    log_stats = (
        data.groupby(grp_keys, sort=False)
            [[f"log_{c}" for c in cols]]
            .agg(['mean', 'std'])
    )
    med_stats = (
        data.groupby(grp_keys, sort=False)
            [cols]
            .median()
            .add_suffix("_median")
    )
    flat = {}
    for c in cols:
        gm   = np.exp(log_stats[(f"log_{c}", "mean")])
        gstd = np.exp(log_stats[(f"log_{c}", "std")])
        flat[f"{c}_gmean"] = gm
        flat[f"{c}_gstd"]  = gstd
    geom_stats = pd.DataFrame(flat, index=log_stats.index)

    res = pd.concat([geom_stats, med_stats], axis=1).reset_index()
    return res

_ST_WINDOWS = [50,100,500]

def _strength_stats_numpy(arr):
    if arr.size & 1:
        arr = arr[:-1]
    bw = arr.reshape(-1,2).T
    d = {}
    for e in _ST_WINDOWS:
        root = "s-score" if e==500 else f"s-score_{e//2}"
        seg = bw[:,:e]
        d[f"{root}_mean"]   = np.nanmean(seg,   axis=1)
        d[f"{root}_median"] = np.nanmedian(seg, axis=1)
        d[f"{root}_std"]    = np.nanstd(seg,    axis=1)
    return d

def _proc_strength(idx,grp):
    PB, PW = grp.iloc[0][['PB','PW']]
    sdict  = _strength_stats_numpy(grp["s-score"].to_numpy(dtype="float32"))

    rows = {
        "Target":      [PB, PW],
        "Opponent":    [PW, PB],
        "id":        [idx, idx],
    }
    for k,v in sdict.items():
        rows[k] = v.tolist()
    return pd.DataFrame(rows)

def compute_strength(df,n_jobs):
    n_jobs = _safe_n_jobs(n_jobs)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_proc_strength)(idx, grp)
        for idx, grp in df.groupby("id", sort=False)
    )
    return pd.concat(parts, ignore_index=True)

_LOSS_WINDOWS = [50, 100, 500]

def _mean_med_std(arr):
    if arr.size == 0 or np.isnan(arr).all():
        return np.nan, np.nan, np.nan
    with np.errstate(all="ignore"):
        return (
            np.nanmean(arr),
            np.nanmedian(arr),
            np.nanstd(arr, ddof=0),
        )

def _loss_stats_numpy(arr_b,arr_w):
    d = {}

    mean_b, med_b, std_b = _mean_med_std(arr_b)
    mean_w, med_w, std_w = _mean_med_std(arr_w)
    d["mean"]   = [mean_b, mean_w]
    d["median"] = [med_b,  med_w]
    d["std"]    = [std_b,  std_w]
    return d

def _proc_loss(idx,grp):
    PB, PW = grp.iloc[0][["PB", "PW"]]
    BR, WR = grp.iloc[0][["BR", "WR"]]
    group,file = grp.iloc[0][['group','file']]

    rows = {
        "Target":       [PB, PW],
        "Opponent":     [PW, PB],
        "TargetRank":    [BR, WR],
        "OpponentRank":  [WR, BR],
        "Color":        ["Black", "White"],
        "file":         [file, file],
        "group":        [group,group],
        'id':           [idx,idx]
    }

    played = grp['loss'].to_numpy(dtype="float32")
    for e in _LOSS_WINDOWS:
        root = "" if e == 500 else f"_{e//2}"
        seg  = played[:e]
        stats = _loss_stats_numpy(seg[::2], seg[1::2])
        for k, v in stats.items():
            rows[f"loss{root}_{k}"] = v
    return pd.DataFrame(rows)

def compute_loss(df,n_jobs):
    n_jobs = _safe_n_jobs(n_jobs)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_proc_loss)(idx,grp) 
        for idx,grp in df.groupby("id", sort=False)
    )
    return pd.concat(parts, ignore_index=True)

def make_game_info(df,start,n_jobs):
    if start is None:
        start = time.time()

    df = _downcast_numeric(df)

    prior   = compute_prior(df)
    print(f"[info] prior     ✅ {time.time()-start:.1f}s")

    strength = compute_strength(df[['id','s-score','PB','PW','BR','WR']], n_jobs)
    print(f"[info] s-score   ✅ {time.time()-start:.1f}s")

    loss    = compute_loss(df, n_jobs)
    print(f"[info] loss      ✅ {time.time()-start:.1f}s")

    print(prior.shape,strength.shape,loss.shape)
    return prior.merge(strength, on=['id','Target']).merge(loss, on=['id','Target','Opponent'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, required=True, help='Target directory to analyze')
    parser.add_argument('-j','--jobs', type=int, default=None,)
    args = parser.parse_args()

    group_list = [
        '3-5k','1-2k','1d','2d','3d','4d','5d','6d','7d',
        '8d','9d'
    ]
    for group in group_list:
        start = time.time()
        base = pd.read_csv(f'{args.target}/{group}_turn_info.csv',low_memory=False)
        
        game_info = make_game_info(base, start, n_jobs=args.jobs)
        end = time.time()
        print('finished making game_info:',format(end-start,'.1f'),'seconds')
        
        output_path = f"{args.target}/{group}_game_info.csv"
        game_info.to_csv(output_path,index=False)
        print(f"[make_game_info] total     ✅ {time.time() - start:.1f}s → {output_path}")
    
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()