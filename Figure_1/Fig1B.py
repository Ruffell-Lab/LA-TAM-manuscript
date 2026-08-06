#Analytical scripts were developed with the assistance of generative AI (Claude, version 1.17377.2). All resulting code was reviewed and deployed under the supervision of study bioinformatician.
import numpy as np, pandas as pd, math, re, matplotlib, openpyxl
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
B="/sessions/keen-modest-newton/mnt/bioinformatics"; DST="/sessions/keen-modest-newton/mnt/Macrophage with ER"
# CIBERSORTx B-mode LM22 fraction outputs (edit to your own run):
CIBX_TCGA =f"{DST}/CIBERSORTx_2026/CIBERSORTx_Job2_Adjusted.txt"   # TCGA ER+
CIBX_SCANB=f"{DST}/CIBERSORTx_2026/CIBERSORTx_Job3_Adjusted.txt"   # SCAN-B ER+
CELLS=["B cells naive","B cells memory","Plasma cells","T cells CD8","T cells CD4 naive",
"T cells CD4 memory resting","T cells CD4 memory activated","T cells follicular helper",
"T cells regulatory (Tregs)","T cells gamma delta","NK cells resting","NK cells activated",
"Monocytes","Macrophages M0","Macrophages M1","Macrophages M2","Dendritic cells resting",
"Dendritic cells activated","Mast cells resting","Mast cells activated","Eosinophils","Neutrophils"]
def mwu(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);a=a[~np.isnan(a)];b=b[~np.isnan(b)];n1,n2=len(a),len(b)
    if n1<3 or n2<3:return np.nan
    allv=np.concatenate([a,b]);r=pd.Series(allv).rank().values;U=r[:n1].sum()-n1*(n1+1)/2
    _,cnt=np.unique(allv,return_counts=True);N=n1+n2;tie=(cnt**3-cnt).sum()
    sd=math.sqrt(n1*n2/12*((N+1)-tie/(N*(N-1))))
    if sd==0:return np.nan
    return math.erfc(abs((U-n1*n2/2)/sd)/math.sqrt(2))
def bh(ps):                       # Benjamini-Hochberg FDR (ignores NaN)
    ps=np.asarray(ps,float); idx=np.where(~np.isnan(ps))[0]; p=ps[idx]; m=len(p)
    out=np.full_like(ps,np.nan)
    if m:
        o=np.argsort(p); q=p[o]*m/(np.arange(m)+1); q=np.minimum.accumulate(q[::-1])[::-1]
        out[idx[o]]=np.clip(q,0,1)
    return out
def st(q): return "" if q!=q else "****" if q<1e-4 else "***" if q<1e-3 else "**" if q<1e-2 else "*" if q<0.05 else ""
def fmt(p):
    if p!=p: return "NA"
    if p>=0.001: return f"{p:.3f}"
    e=int(np.floor(np.log10(p))); return f"{p/10**e:.0f}e{e}"
def n15(x):return str(x).replace(".","-")[:15]
# ---- TCGA (CIBERSORTx B-mode) ----
cib=pd.read_csv(CIBX_TCGA,sep="\t");cib["sample"]=cib["Mixture"].map(n15)
cm=pd.read_csv(f"{B}/TCGA/2025/cbio_brca (TCGA)/TCGA_BRCA_clinicalMatrix.tsv",sep="\t",low_memory=False)
idc=[c for c in ["sampleID","Sample"] if c in cm.columns][0];cm["sample"]=cm[idc].map(n15)
erc="ER_Status_nature2012" if "ER_Status_nature2012" in cm.columns else "breast_carcinoma_estrogen_receptor_status"
erpos=set(cm.loc[cm[erc].astype(str).str.upper()=="POSITIVE","sample"])
sv=pd.read_csv(f"{B}/TCGA/2025/cbio_brca (TCGA)/TCGA_PanCan_Survival.tsv",sep="\t",low_memory=False);sv["sample"]=sv["sample"].map(n15)
tx=pd.read_csv(f"{B}/brca_tcga_pan_can_atlas_2018 3/data_timeline_treatment.txt",sep="\t",low_memory=False)
ht=set(tx.loc[tx.TREATMENT_TYPE=="Hormone Therapy","PATIENT_ID"])
T=cib[cib["sample"].isin(erpos)].merge(sv[["sample","PFI"]],on="sample");T=T[T["sample"].str[:12].isin(ht)];T["PFI"]=pd.to_numeric(T["PFI"],errors="coerce")
Tg={"No relapse":T[T.PFI==0],"Relapsed":T[T.PFI==1]}
# ---- SCAN-B (CIBERSORTx B-mode) ----
sc=pd.read_csv(CIBX_SCANB,sep="\t");sc["F"]=sc["Mixture"].astype(str)
lab=pd.read_csv(f"{DST}/SCANB_relapse_macro_efflux.csv")[["F","ERpos","endo","RFi_e"]].drop_duplicates("F")
sc=sc.merge(lab,on="F",how="inner")
sc=sc[(sc["ERpos"]==True)&(sc["endo"]==True)]
sc["RFi_e"]=pd.to_numeric(sc["RFi_e"],errors="coerce")
Sg={"No relapse":sc[sc.RFi_e==0],"Relapsed":sc[sc.RFi_e==1]}
print("TCGA endo PFI no/yes:",len(Tg["No relapse"]),len(Tg["Relapsed"]),"| SCAN-B endo RFi no/yes:",len(Sg["No relapse"]),len(Sg["Relapsed"]))
cols=[("TCGA","No relapse"),("TCGA","Relapsed"),("SCAN-B","No relapse"),("SCAN-B","Relapsed")]
M=np.zeros((len(CELLS),4))
for j,(coh,grp) in enumerate(cols):
    d=(Tg if coh=="TCGA" else Sg)[grp];M[:,j]=[d[c].mean() for c in CELLS]
Mz=np.zeros_like(M)
for blk in [slice(0,2),slice(2,4)]:
    sub=M[:,blk];Mz[:,blk]=(sub-sub.mean(1,keepdims=True))/(sub.std(1,keepdims=True)+1e-9)
# per-cohort MWU p and BH FDR (family = 22 cells)
P={};Q={}
for coh,g in [("TCGA",Tg),("SCAN-B",Sg)]:
    ps=np.array([mwu(g["No relapse"][c],g["Relapsed"][c]) for c in CELLS],float)
    P[coh]=ps;Q[coh]=bh(ps)
fig,ax=plt.subplots(figsize=(7.4,9.2))
im=ax.imshow(Mz,cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2),aspect="auto")
ncol=[len((Tg if c=="TCGA" else Sg)[g]) for c,g in cols]
ax.set_xticks(range(4));ax.set_xticklabels([f"{g}\n(n={n})" for (_,g),n in zip(cols,ncol)],fontsize=8)
ax.set_yticks(range(len(CELLS)));ax.set_yticklabels(CELLS,fontsize=9)
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(CELLS),1),minor=True)
ax.grid(which="minor",color="white",lw=1.4);ax.tick_params(which="minor",length=0)
relcol={"TCGA":cols.index(("TCGA","Relapsed")),"SCAN-B":cols.index(("SCAN-B","Relapsed"))}
for coh in ("TCGA","SCAN-B"):
    j=relcol[coh];ps=P[coh];q=Q[coh]
    for i in range(len(CELLS)):
        sig=(q[i]==q[i] and q[i]<0.05);col="black" if sig else "#666666";fw="bold" if sig else "normal"
        ax.text(j,i-0.16,f"p {fmt(ps[i])}",ha="center",va="center",fontsize=5.3,color=col,fontweight=fw)
        ax.text(j,i+0.20,f"FDR {fmt(q[i])}",ha="center",va="center",fontsize=5.3,color=col,fontweight=fw)
        s=st(q[i])
        if s:ax.text(j+0.44,i-0.30,s,ha="right",va="center",fontsize=7,color="black")
ax.axvline(1.5,color="white",lw=3)
ax.text(0.5,-0.95,f"TCGA (n={len(T)})",ha="center",fontweight="bold",fontsize=9)
ax.text(2.5,-0.95,f"SCAN-B (n={len(sc)})",ha="center",fontweight="bold",fontsize=9)
for i,c in enumerate(CELLS):
    if c.startswith("Macrophages M1") or c.startswith("Macrophages M2"):
        ax.add_patch(plt.Rectangle((-0.5,i-0.5),4,1,fill=False,edgecolor="#e67e22",lw=1.8))
cb=fig.colorbar(im,ax=ax,shrink=0.35,pad=0.02);cb.set_label("Row-scaled\nimmune fraction",fontsize=8)
ax.set_title("Baseline immune profile by eventual outcome\nwithin ENDOCRINE-TREATED ER+ (relapsed vs not)  |  CIBERSORTx B-mode",fontsize=9.5,pad=30)
fig.text(0.5,0.045,"Relapsed columns: p = Mann-Whitney U (relapsed vs no-relapse); FDR = Benjamini-Hochberg (22 cells). "
         "Bold = FDR<0.05. Stars from FDR. Row-scaled within cohort.",ha="center",fontsize=6.4,color="#555")
fig.subplots_adjust(bottom=0.12,top=0.9)
plt.savefig(f"{DST}/Immune_heatmap_endocrine_byRelapse_TCGA_SCANB.pdf",bbox_inches="tight");print("saved")
