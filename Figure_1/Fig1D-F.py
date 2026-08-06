#Analytical scripts were developed with the assistance of generative AI (Claude, version 1.17377.2). All resulting code was reviewed and deployed under the supervision of study bioinformatician.
# Figure 1D-F (SCAN-B) — macrophage burden vs survival, KM + log-rank.
# Survival comparisons are pre-specified/confirmatory -> raw log-rank P (no FDR).
import numpy as np, pandas as pd, math, re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
B="/sessions/keen-modest-newton/mnt/bioinformatics"; DST="/sessions/keen-modest-newton/mnt/Macrophage with ER"
# CIBERSORTx B-mode LM22 fractions for SCAN-B ER+ (edit to your own output):
CIBX_SCANB=f"{DST}/CIBERSORTx_2026/CIBERSORTx_Job3_Adjusted.txt"

# ---------- stats ----------
def logrank(t,e,g):
    d=pd.DataFrame({'t':t,'e':e,'g':g}).dropna(); O=E=V=0.0
    for tt in np.sort(d.loc[d.e==1,'t'].unique()):
        ar=d[d.t>=tt];n=len(ar);n1=(ar.g==1).sum();de=d[(d.t==tt)&(d.e==1)];dn=len(de);d1=(de.g==1).sum()
        if n>1:E+=dn*n1/n;V+=dn*(n1/n)*(1-n1/n)*(n-dn)/(n-1);O+=d1
    if V<=0: return 1.0,0.0
    z=(O-E)/math.sqrt(V); return math.erfc(abs(z)/math.sqrt(2)), z
def km(t,e,scale):
    d=pd.DataFrame({'t':t,'e':e}).dropna().sort_values('t');ts=[0];sv=[1.];c=1
    for tt in np.sort(d.loc[d.e==1,'t'].unique()):
        n=(d.t>=tt).sum();dd=((d.t==tt)&(d.e==1)).sum();c*=(1-dd/n);ts.append(tt/scale);sv.append(c)
    return ts,sv
def best_cut(x,t,e):
    xs=x.values; cand=np.unique(np.nanpercentile(xs,np.linspace(20,80,20))); bestz=-1;bestc=np.nanmedian(xs)
    for c in cand:
        g=(xs>c).astype(int)
        if g.sum()<10 or (len(g)-g.sum())<10: continue
        p,z=logrank(t,e,g)
        if abs(z)>bestz: bestz=abs(z);bestc=c
    return bestc

def panel(ax,df,score,tcol,ecol,scale,method,ylab):
    s=df.dropna(subset=[score,tcol,ecol]); s=s[s[tcol]>0]
    x=s[score]
    cut=x.median() if method=="median" else best_cut(x,s[tcol].values,s[ecol].values)
    hi=(x>cut).astype(int); p,_=logrank(s[tcol].values,s[ecol].values,hi.values)
    for h,c,nm in [(1,"#c0392b","high"),(0,"#3366cc","low")]:
        g=s[hi==h]; ts,sv=km(g[tcol].values,g[ecol].values,scale)
        ax.step(ts,sv,where="post",color=c,lw=1.6,label=f"{nm} (n={len(g)})")
    ax.set_ylim(0,1.02); ax.set_xlabel("years"); ax.set_ylabel(ylab); ax.legend(fontsize=7,loc="lower left",frameon=False)
    star=" *" if p<0.05 else ""
    ax.set_title(f"{score}\nlog-rank p={p:.2g}{star}",fontsize=10)

SIG=[("macro","Total macrophage"),("M0","M0-like"),("M1","M1-like"),("M2","M2-like")]
def page(pdf,df,tcol,ecol,scale,method,suptitle,note=""):
    fig,axs=plt.subplots(1,4,figsize=(17,4.3))
    for ax,(col,lab) in zip(axs,SIG):
        panel(ax,df,col,tcol,ecol,scale,method,"survival" if ax is axs[0] else "")
        ax.set_title(ax.get_title().replace(col,lab),fontsize=10)
    sup=suptitle+("   [best-cutoff: p optimistic — use median for inference]" if method=="bestcut" else "")
    fig.suptitle(sup,fontweight="bold",fontsize=11,y=1.05)
    if note: fig.text(0.5,-0.02,note,ha="center",fontsize=7,color="#555")
    plt.tight_layout(); pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)

# ================= SCAN-B =================
s=pd.read_csv(f"{DST}/SCANB_relapse_macro_efflux.csv")
s=s[s["ERpos"]==True]
# replace the classic-CIBERSORT macro/M0/M1/M2 columns with CIBERSORTx B-mode fractions
cib=pd.read_csv(CIBX_SCANB,sep="\t"); cib["F"]=cib["Mixture"].astype(str)
cib["macro"]=cib[["Macrophages M0","Macrophages M1","Macrophages M2"]].sum(1)
cib["M0"]=cib["Macrophages M0"]; cib["M1"]=cib["Macrophages M1"]; cib["M2"]=cib["Macrophages M2"]
s=s.drop(columns=[c for c in ["macro","M0","M1","M2"] if c in s.columns]).merge(
    cib[["F","macro","M0","M1","M2"]],on="F",how="left")
print("SCAN-B ER+:",len(s),"endo:",s['endo'].sum())
with PdfPages(f"{DST}/SCANB_KM_panels.pdf") as pdf:
    for ep,(td,ee) in [("OS",("OS_d","OS_e")),("RFi",("RFi_d","RFi_e")),("DRFi",("DRFi_d","DRFi_e"))]:
        page(pdf,s,td,ee,365.25,"median",f"SCAN-B ER+ — {ep}, median split (n={s[td].notna().sum()})")
        page(pdf,s,td,ee,365.25,"bestcut",f"SCAN-B ER+ — {ep}, best-cutoff")
        page(pdf,s[s.endo],td,ee,365.25,"median",f"SCAN-B ER+ ENDOCRINE-treated — {ep}, median (n={s[s.endo][td].notna().sum()})")
        page(pdf,s[~s.endo],td,ee,365.25,"median",f"SCAN-B ER+ UNTREATED — {ep}, median (n={s[~s.endo][td].notna().sum()})")
print("saved SCANB_KM_panels.pdf")
