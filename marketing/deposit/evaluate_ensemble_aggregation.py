"""Does averaging the ensemble destroy a hypo signal that individual models have?"""
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
paths=sorted(glob.glob(os.path.join(REPO,"Models","Modern Models","*.onnx")))
sess=[ort.InferenceSession(p) for p in paths]
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
def rec(pred,th=70):
    tp=((A<th)&(pred<th)).sum(); fn=((A<th)&(pred>=th)).sum()
    fp=((A>=th)&(pred<th)).sum()
    return tp,fn,(tp/(tp+fn)*100 if tp+fn else 0),(tp/(tp+fp)*100 if tp+fp else 0)
print(f"  {len(rows)} rollouts, {int((A<70).sum())} true hypo events (<70 mg/dL)\n")
print(f"  {'strategy':<30}{'caught':>8}{'missed':>8}{'recall':>9}{'precision':>11}")
print("  "+"-"*66)
for k,p in enumerate(paths):
    tp,fn,r_,pr=rec(per[k])
    print(f"  {'single: '+os.path.basename(p)[-11:]:<30}{tp:>8}{fn:>8}{r_:>8.1f}%{pr:>10.1f}%")
print("  "+"-"*66)
for lab,pred in [("ensemble MEAN (shipped)",per.mean(0)),
                 ("ensemble MIN (most cautious)",per.min(0)),
                 ("any model says low",np.where(per.min(0)<70,per.min(0),per.mean(0)))]:
    tp,fn,r_,pr=rec(pred)
    print(f"  {lab:<30}{tp:>8}{fn:>8}{r_:>8.1f}%{pr:>10.1f}%")
