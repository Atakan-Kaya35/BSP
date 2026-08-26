"""Aggregate error hides event performance. Measure the cases the product exists for."""
import numpy as np, pandas as pd, onnxruntime as ort, glob, os, warnings
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
P=[];A=[];L=[]
for r in rows:
    cur=Xt[r:r+1].copy(); clock=tod[base+r-1]
    for h in range(3):
        p=float(np.mean([s.run([o],{k:cur.astype(np.float32)})[0][0][0] for s,k,o in zip(sess,nm,om)]))
        clock=(clock+5)%1440
        cur=np.append(cur[:,1:,:],[np.array([[p,(np.sin(2*np.pi*clock/1440)+1)/2,(np.cos(2*np.pi*clock/1440)+1)/2]])],axis=1)
    P.append(inv(p)); A.append(raw[base+r+2]); L.append(inv(Xt[r,-1,0]))
P,A,L=np.array(P),np.array(A),np.array(L); move=A-L
print(f"test rollouts: {len(P):,}   (+15 min horizon)\n")
print(f"  {'window type':<34}{'n':>6}{'MAE':>8}{'RMSE':>8}")
print("  "+"-"*56)
for lab,mask in [("all windows",np.ones(len(P),bool)),
                 ("quiet   (|move| <= 10 mg/dL)",np.abs(move)<=10),
                 ("moving  (|move| > 20 mg/dL)",np.abs(move)>20),
                 ("falling fast (move < -20)",move<-20),
                 ("rising fast  (move > +20)",move>20)]:
    if mask.sum()==0: continue
    e=P[mask]-A[mask]
    print(f"  {lab:<34}{mask.sum():>6}{np.mean(np.abs(e)):>8.1f}{np.sqrt(np.mean(e**2)):>8.1f}")
print(f"\n  HYPOGLYCEMIA DETECTION (+15 min)")
print("  "+"-"*56)
for th in (70,80):
    tp=int(((A<th)&(P<th)).sum()); fn=int(((A<th)&(P>=th)).sum())
    fp=int(((A>=th)&(P<th)).sum())
    rec=tp/(tp+fn) if tp+fn else float('nan')
    prec=tp/(tp+fp) if tp+fp else float('nan')
    print(f"  actual < {th} mg/dL: {tp+fn:>3} events   caught {tp:>3}   missed {fn:>3}"
          f"   recall {rec*100:>5.1f}%   precision {(prec*100 if tp+fp else 0):>5.1f}%")
print(f"\n  share of test windows that are quiet (|move|<=10): {np.mean(np.abs(move)<=10)*100:.0f}%")
