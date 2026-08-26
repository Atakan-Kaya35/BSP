"""One BSP training experiment. Same split and preprocessing as the production
Training Container; the variable is what we ask the network to learn."""
import os, sys, json, argparse, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("--name",required=True)
ap.add_argument("--target",default="abs",choices=["abs","delta","direct3","clf"])
ap.add_argument("--loss",default="mse",choices=["mse","weighted","asym","bce"])
ap.add_argument("--cell",default="lstm",choices=["lstm","gru"])
ap.add_argument("--seq",type=int,default=12)
ap.add_argument("--units",type=int,default=50)
ap.add_argument("--layers",type=int,default=3)
ap.add_argument("--dropout",type=float,default=0.2)
ap.add_argument("--epochs",type=int,default=60)
ap.add_argument("--batch",type=int,default=24)
ap.add_argument("--lr",type=float,default=1e-3)
ap.add_argument("--asym_alpha",type=float,default=3.0)
ap.add_argument("--seeds",type=int,default=3)
ap.add_argument("--threads",type=int,default=4)
A=ap.parse_args()

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(A.threads)
tf.config.threading.set_inter_op_parallelism_threads(max(1,A.threads//2))
from tensorflow import keras
from tensorflow.keras import layers as L

REPO=r"C:\Users\atakan\Documents\BSP\BSP-repo"; SKIP=25
TIME="Zaman damgası (GG-AA-YYYY/ss:dd:sn)"; GLU="Glikoz Değeri (mg/dL)"
LOW,HIGH,HYPO=80,200,70

df=pd.read_csv(os.path.join(REPO,"atakanka350@gmail.com.csv"),sep=";")
df.iloc[SKIP:,7]=df.iloc[SKIP:,7].replace({"Yüksek":"400","Düşük":"40"})
df[GLU]=pd.to_numeric(df[GLU],errors="coerce"); df[TIME]=pd.to_datetime(df[TIME],errors="coerce")
mn=df[TIME].dt.hour*60+df[TIME].dt.minute
df["s"],df["c"]=np.sin(2*np.pi*mn/1440),np.cos(2*np.pi*mn/1440)
df=df.dropna(subset=[GLU,"s","c"]).iloc[SKIP:]
g=(df[[GLU]].values-39.)/361.; st=(df[["s"]].values+1)/2; ct=(df[["c"]].values+1)/2
data=np.hstack((g,st,ct)); raw=df[GLU].values
S=A.seq; inv=lambda v:v*361.+39.

X=np.array([data[i-S:i] for i in range(S,len(data)-2)])
y1=np.array([data[i,0] for i in range(S,len(data)-2)])                       # +5
y3=np.array([data[i+2,0] for i in range(S,len(data)-2)])                     # +15
Y3=np.array([data[i:i+3,0] for i in range(S,len(data)-2)])                   # all three
last=X[:,-1,0]

nte=int(len(X)*0.10); ntr=len(X)-nte; nva=int(ntr*0.10)
sl_tr=slice(0,ntr-nva); sl_va=slice(ntr-nva,ntr); sl_te=slice(ntr,len(X))

def targets(mode):
    if mode=="abs":     return y1,        None
    if mode=="delta":   return y1-last,   None
    if mode=="direct3": return Y3,        None
    if mode=="clf":     return (inv(y3)<LOW).astype("float32"), None
    raise ValueError(mode)
Y,_=targets(A.target)

w=None
if A.loss=="weighted":
    mv=np.abs(inv(y1)-inv(last)); w=1.0+(mv/np.maximum(mv.std(),1e-6))
def asym(alpha):
    def f(yt,yp):
        e=yt-yp
        return tf.reduce_mean(tf.where(e<0, alpha*tf.square(e), tf.square(e)))
    return f                                    # e<0 means over-prediction: missed a fall

def build(seed):
    tf.keras.utils.set_random_seed(seed)
    Cell=L.LSTM if A.cell=="lstm" else L.GRU
    m=keras.Sequential(name=A.name.replace("-","_"))
    m.add(L.Input(shape=(S,3)))
    for i in range(A.layers-1):
        m.add(Cell(A.units,return_sequences=True)); m.add(L.Dropout(A.dropout))
    m.add(Cell(A.units,return_sequences=False)); m.add(L.Dropout(A.dropout))
    if A.target=="direct3": m.add(L.Dense(3))
    elif A.target=="clf":   m.add(L.Dense(1,activation="sigmoid"))
    else:                   m.add(L.Dense(1))
    lo={"mse":"mse","weighted":"mse","asym":asym(A.asym_alpha),"bce":"binary_crossentropy"}[A.loss]
    m.compile(optimizer=keras.optimizers.Adam(A.lr),loss=lo)
    return m

def forecast15(m,Xs):
    """+15 min prediction in mg/dL for a batch, honouring the target mode."""
    if A.target=="direct3": return inv(m.predict(Xs,verbose=0)[:,2])
    if A.target=="clf":     return m.predict(Xs,verbose=0).ravel()
    cur=Xs.copy()
    for _ in range(3):
        p=m.predict(cur,verbose=0).ravel()
        nxt=p if A.target=="abs" else cur[:,-1,0]+p
        nxt=np.clip(nxt,0,1)
        step=np.stack([nxt,cur[:,-1,1],cur[:,-1,2]],axis=1)[:,None,:]
        cur=np.concatenate([cur[:,1:,:],step],axis=1)
    return inv(nxt)

preds=[]
for sd in range(A.seeds):
    m=build(1000+sd)
    es=keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True)
    kw={"sample_weight":w[sl_tr]} if w is not None else {}
    m.fit(X[sl_tr],Y[sl_tr],validation_data=(X[sl_va],Y[sl_va]),epochs=A.epochs,
          batch_size=A.batch,callbacks=[es],verbose=0,**kw)
    preds.append(forecast15(m,X[sl_te]))
P=np.mean(preds,axis=0); Pmin=np.min(preds,axis=0)
act=inv(y3[sl_te]); lastte=inv(last[sl_te]); move=act-lastte
out={"name":A.name,"config":vars(A),"n_test":int(len(act)),"seeds":A.seeds}

if A.target=="clf":
    for th in (0.3,0.5):
        pos=P>=th; t=act<LOW
        tp=int((t&pos).sum()); fn=int((t&~pos).sum()); fp=int((~t&pos).sum())
        out[f"lowwarn@{th}"]={"recall":round(tp/(tp+fn)*100,1) if tp+fn else 0,
                              "precision":round(tp/(tp+fp)*100,1) if tp+fp else 0,"tp":tp,"fn":fn,"fp":fp}
else:
    out["mae_all"]=round(float(np.mean(np.abs(P-act))),2)
    out["rmse_all"]=round(float(np.sqrt(np.mean((P-act)**2))),2)
    q=np.abs(move)<=10; mo=np.abs(move)>20
    out["mae_quiet"]=round(float(np.mean(np.abs(P[q]-act[q]))),2) if q.sum() else None
    out["mae_moving"]=round(float(np.mean(np.abs(P[mo]-act[mo]))),2) if mo.sum() else None
    for lab,pred in [("mean",P),("min",Pmin)]:
        t=act<HYPO; pos=pred<HYPO
        tp=int((t&pos).sum()); fn=int((t&~pos).sum()); fp=int((~t&pos).sum())
        out[f"hypo70_{lab}"]={"recall":round(tp/(tp+fn)*100,1) if tp+fn else 0,
                              "precision":round(tp/(tp+fp)*100,1) if tp+fp else 0,"tp":tp,"fn":fn}
        t2=(act<LOW)|(act>HIGH); pos2=(pred<LOW)|(np.max(preds,axis=0)>HIGH if lab=="min" else pred>HIGH)
        tp2=int((t2&pos2).sum()); fn2=int((t2&~pos2).sum()); fp2=int((~t2&pos2).sum())
        r=tp2/(tp2+fn2)*100 if tp2+fn2 else 0; pr=tp2/(tp2+fp2)*100 if tp2+fp2 else 0
        out[f"oor_{lab}"]={"recall":round(r,1),"precision":round(pr,1),
                           "f1":round(2*r*pr/(r+pr),1) if r+pr else 0}
d=os.path.join(os.path.dirname(__file__),"results"); os.makedirs(d,exist_ok=True)
# dump raw per-seed predictions so threshold/aggregation sweeps need no retraining
np.savez(os.path.join(d,f"{A.name}_preds.npz"),
         preds=np.array(preds), actual=act, last=lastte)
json.dump(out,open(os.path.join(d,f"{A.name}.json"),"w"),indent=1,default=str)
print(json.dumps({k:v for k,v in out.items() if k!="config"},indent=1))
