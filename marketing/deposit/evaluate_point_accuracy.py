"""Held-out eval, rolling forward the REAL clock exactly as the inference container does."""
import numpy as np, pandas as pd, onnxruntime as ort, glob, os, warnings
warnings.filterwarnings("ignore")
REPO = r"C:\Users\atakan\Documents\BSP\BSP-repo"
SKIP, SEQ, TEST = 25, 12, 0.10
TIME_COL = "Zaman damgası (GG-AA-YYYY/ss:dd:sn)"; GLU = "Glikoz Değeri (mg/dL)"

df = pd.read_csv(os.path.join(REPO, "atakanka350@gmail.com.csv"), sep=";")
df.iloc[SKIP:, 7] = df.iloc[SKIP:, 7].replace({"Yüksek": "400", "Düşük": "40"})
df[GLU] = pd.to_numeric(df[GLU], errors="coerce")
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
mins = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute
df["sin_time"], df["cos_time"] = np.sin(2*np.pi*mins/1440), np.cos(2*np.pi*mins/1440)
df["mins"] = mins
df = df.dropna(subset=[GLU, "sin_time", "cos_time"]).iloc[SKIP:]

data = np.hstack(((df[[GLU]].values - 39.)/361., (df[["sin_time"]].values+1)/2, (df[["cos_time"]].values+1)/2))
raw, tod = df[GLU].values, df["mins"].values
X = np.array([data[i-SEQ:i] for i in range(SEQ, len(data))])
n_test = int(len(X)*TEST); Xt = X[-n_test:]; base = len(data) - n_test

sess = [ort.InferenceSession(p) for p in sorted(glob.glob(os.path.join(REPO,"Models","Modern Models","*.onnx")))]
nm = [s.get_inputs()[0].name for s in sess]; om = [s.get_outputs()[0].name for s in sess]
inv = lambda v: v*361.+39.

STEP = max(1, n_test//400); rows = [r for r in range(0, n_test-3, STEP) if base+r+2 < len(raw)]
pred_h = {1: [], 2: [], 3: []}
for r in rows:
    cur = Xt[r:r+1].copy()
    clock = tod[base + r - 1]                       # real time of the last observed reading
    for h in (1, 2, 3):
        p = float(np.mean([s.run([o], {n: cur.astype(np.float32)})[0][0][0] for s,n,o in zip(sess,nm,om)]))
        pred_h[h].append(inv(p))
        clock = (clock + 5) % 1440                   # advance the clock, as inference does
        cur = np.append(cur[:, 1:, :], [np.array([[p,
                (np.sin(2*np.pi*clock/1440)+1)/2, (np.cos(2*np.pi*clock/1440)+1)/2]])], axis=1)

last = np.array([inv(Xt[r,-1,0]) for r in rows]); prev = np.array([inv(Xt[r,-2,0]) for r in rows])
print(f"readings {len(df):,}   test rollouts {len(rows):,}   models ensembled {len(sess)}\n")
print(f"  {'horizon':<10}{'method':<26}{'MAE':>7}{'RMSE':>8}{'MARD':>8}")
print("  " + "-"*59)
for h, lab, mult in [(1,"+5 min",1), (2,"+10 min",2), (3,"+15 min",3)]:
    act = np.array([raw[base+r+h-1] for r in rows])
    cands = [("BSP ensemble", np.array(pred_h[h])),
             ("persistence (no change)", last),
             ("linear extrapolation", last + (last-prev)*mult)]
    for i,(name,pr) in enumerate(cands):
        e = pr - act
        print(f"  {lab if i==0 else '':<10}{name:<26}{np.mean(np.abs(e)):>7.1f}"
              f"{np.sqrt(np.mean(e**2)):>8.1f}{np.mean(np.abs(e/act))*100:>7.1f}%")
    print()
