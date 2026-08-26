"""Evaluate the safeness flag as a binary warning: 'will you be out of [80,200] at +15?'"""
import numpy as np, pandas as pd, onnxruntime as ort, glob, os, warnings, json
warnings.filterwarnings("ignore")
REPO=r"C:\Users\atakan\Documents\BSP\BSP-repo"; SKIP,SEQ,TEST=25,12,0.10
TIME="Zaman damgası (GG-AA-YYYY/ss:dd:sn)"; GLU="Glikoz Değeri (mg/dL)"
df=pd.read_csv(os.path.join(REPO,"atakanka350@gmail.com.csv"),sep=";")
df.iloc[SKIP:,7]=df.iloc[SKIP:,7].replace({"Yüksek":"400","Düşük":"40"})
df[GLU]=pd.to_numeric(df[GLU],errors="coerce"); df[TIME]=pd.to_datetime(df[TIME],errors="coerce")
m=df[TIME].dt.hour*60+df[TIME].dt.minute
df["s"],df["c"]=np.sin(2*np.pi*m/1440),np.cos(2*np.pi*m/1440); df["m"]=m
df=df.dropna(subset=[GLU,"s","c"]).iloc[SKIP:]
data=np.hstack(((df[[GLU]].values-39.)/361.,(df[["s"]].values+1)/2,(df[["c"]].values+1)/2))
raw,tod=df[GLU].values,df["m"].values
X=np.array([data[i-SEQ:i] for i in range(SEQ,len(data))])
n=int(len(X)*TEST); Xt=X[-n:]; base=len(data)-n
sess=[ort.InferenceSession(p) for p in sorted(glob.glob(os.path.join(REPO,"Models","Modern Models","*.onnx")))]
nm=[s.get_inputs()[0].name for s in sess]; om=[s.get_outputs()[0].name for s in sess]
inv=lambda v:v*361.+39.
rows=[r for r in range(0,n-3,3) if base+r+2<len(raw)]
per=np.zeros((len(sess),len(rows))); A=np.zeros(len(rows))
for j,r in enumerate(rows):
    A[j]=raw[base+r+2]
    for k,(s,kk,o) in enumerate(zip(sess,nm,om)):
        cur=Xt[r:r+1].copy(); clock=tod[base+r-1]
        for h in range(3):
            p=float(s.run([o],{kk:cur.astype(np.float32)})[0][0][0]); clock=(clock+5)%1440
            cur=np.append(cur[:,1:,:],[np.array([[p,(np.sin(2*np.pi*clock/1440)+1)/2,(np.cos(2*np.pi*clock/1440)+1)/2]])],axis=1)
        per[k,j]=inv(p)
np.save(os.path.join(os.path.dirname(__file__),"per_model_preds.npy"), per)
np.save(os.path.join(os.path.dirname(__file__),"actuals.npy"), A)
LOW,HIGH=80,200
truth = (A<LOW)|(A>HIGH)
def report(lab, unsafe):
    tp=int((truth&unsafe).sum()); fn=int((truth&~unsafe).sum()); fp=int((~truth&unsafe).sum()); tn=int((~truth&~unsafe).sum())
    rec=tp/(tp+fn)*100 if tp+fn else 0; pre=tp/(tp+fp)*100 if tp+fp else 0
    f1=2*rec*pre/(rec+pre) if rec+pre else 0
    print(f"  {lab:<34}{tp:>5}{fn:>7}{fp:>6}{rec:>9.1f}%{pre:>11.1f}%{f1:>7.1f}")
print(f"  {len(rows)} rollouts, {int(truth.sum())} genuinely out-of-range at +15 min\n")
print(f"  {'safeness rule':<34}{'TP':>5}{'FN':>7}{'FP':>6}{'recall':>10}{'precision':>11}{'F1':>7}")
print("  "+"-"*80)
report("mean < 80 or mean > 200  (old)", (per.mean(0)<LOW)|(per.mean(0)>HIGH))
report("min < 80 or max > 200    (new)", (per.min(0)<LOW)|(per.max(0)>HIGH))
for q in (10,20,30):
    lo=np.percentile(per,q,axis=0); hi=np.percentile(per,100-q,axis=0)
    report(f"p{q} < 80 or p{100-q} > 200", (lo<LOW)|(hi>HIGH))
