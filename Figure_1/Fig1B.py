# immune_heatmap_endocrine.py
# Baseline LM22 immune profile by eventual outcome, within endocrine-treated ER+ disease.
# Two cohorts side by side: TCGA (PFI) and SCAN-B (RFi). Fractions from CIBERSORTx B-mode.
# Analytical scripts were developed with the assistance of generative AI
# (Claude, version 1.17377.2). All resulting code was reviewed and deployed
# under the supervision of the study bioinformatician.

import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


BIO_DIR = "/sessions/keen-modest-newton/mnt/bioinformatics"
DST = "/sessions/keen-modest-newton/mnt/Macrophage with ER"

# CIBERSORTx B-mode LM22 fraction outputs - edit to point at your own run
CIBX_TCGA = DST + "/CIBERSORTx_2026/CIBERSORTx_Job2_Adjusted.txt"    # TCGA ER+
CIBX_SCANB = DST + "/CIBERSORTx_2026/CIBERSORTx_Job3_Adjusted.txt"   # SCAN-B ER+

TCGA_DIR = BIO_DIR + "/TCGA/2025/cbio_brca (TCGA)"
CLINICAL_MATRIX = TCGA_DIR + "/TCGA_BRCA_clinicalMatrix.tsv"
SURVIVAL = TCGA_DIR + "/TCGA_PanCan_Survival.tsv"
TREATMENT = BIO_DIR + "/brca_tcga_pan_can_atlas_2018 3/data_timeline_treatment.txt"

SCANB_LABELS = DST + "/SCANB_relapse_macro_efflux.csv"
OUT_PDF = DST + "/Immune_heatmap_endocrine_byRelapse_TCGA_SCANB.pdf"

CELLS = [
    "B cells naive", "B cells memory", "Plasma cells",
    "T cells CD8", "T cells CD4 naive", "T cells CD4 memory resting",
    "T cells CD4 memory activated", "T cells follicular helper",
    "T cells regulatory (Tregs)", "T cells gamma delta",
    "NK cells resting", "NK cells activated",
    "Monocytes", "Macrophages M0", "Macrophages M1", "Macrophages M2",
    "Dendritic cells resting", "Dendritic cells activated",
    "Mast cells resting", "Mast cells activated",
    "Eosinophils", "Neutrophils",
]


def mannwhitney_p(a, b):
    """Two-sided Mann-Whitney U, normal approximation with tie correction.

    Returns NaN if either group has fewer than 3 usable values.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return np.nan

    pooled = np.concatenate([a, b])
    ranks = pd.Series(pooled).rank().values
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2

    _, counts = np.unique(pooled, return_counts=True)
    n = n1 + n2
    ties = (counts ** 3 - counts).sum()

    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    if sd == 0:
        return np.nan

    z = abs(u - n1 * n2 / 2) / sd
    return math.erfc(z / math.sqrt(2))


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values; NaNs pass through as NaN."""
    p_in = np.asarray(pvals, dtype=float)
    keep = np.where(~np.isnan(p_in))[0]
    p = p_in[keep]
    m = len(p)

    out = np.full_like(p_in, np.nan)
    if m == 0:
        return out

    order = np.argsort(p)
    q = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[keep[order]] = np.clip(q, 0, 1)
    return out


def star(q):
    if q != q:          
        return ""
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def fmt_p(p):
    """Three decimals down to 0.001, scientific notation below that."""
    if p != p:
        return "NA"
    if p >= 0.001:
        return "%.3f" % p
    exp = int(np.floor(np.log10(p)))
    return "%.0fe%d" % (p / 10 ** exp, exp)


def short_id(x):
    """TCGA barcodes across these files use '.' vs '-' inconsistently."""
    return str(x).replace(".", "-")[:15]


# TCGA, ER+, endocrine

tcga_cibx = pd.read_csv(CIBX_TCGA, sep="\t")
tcga_cibx["sample"] = tcga_cibx["Mixture"].map(short_id)

clin = pd.read_csv(CLINICAL_MATRIX, sep="\t", low_memory=False)
id_col = [c for c in ["sampleID", "Sample"] if c in clin.columns][0]
clin["sample"] = clin[id_col].map(short_id)

er_col = ("ER_Status_nature2012" if "ER_Status_nature2012" in clin.columns
          else "breast_carcinoma_estrogen_receptor_status")
er_positive = set(clin.loc[clin[er_col].astype(str).str.upper() == "POSITIVE", "sample"])

surv = pd.read_csv(SURVIVAL, sep="\t", low_memory=False)
surv["sample"] = surv["sample"].map(short_id)

treat = pd.read_csv(TREATMENT, sep="\t", low_memory=False)
hormone_therapy = set(treat.loc[treat.TREATMENT_TYPE == "Hormone Therapy", "PATIENT_ID"])

tcga = tcga_cibx[tcga_cibx["sample"].isin(er_positive)].merge(
    surv[["sample", "PFI"]], on="sample")
tcga = tcga[tcga["sample"].str[:12].isin(hormone_therapy)]
tcga["PFI"] = pd.to_numeric(tcga["PFI"], errors="coerce")

tcga_groups = {"No relapse": tcga[tcga.PFI == 0],
               "Relapsed": tcga[tcga.PFI == 1]}


# SCAN-B, ER+, endocrine

scanb = pd.read_csv(CIBX_SCANB, sep="\t")
scanb["F"] = scanb["Mixture"].astype(str)

labels = pd.read_csv(SCANB_LABELS)[["F", "ERpos", "endo", "RFi_e"]].drop_duplicates("F")
scanb = scanb.merge(labels, on="F", how="inner")
scanb = scanb[(scanb["ERpos"] == True) & (scanb["endo"] == True)]
scanb["RFi_e"] = pd.to_numeric(scanb["RFi_e"], errors="coerce")

scanb_groups = {"No relapse": scanb[scanb.RFi_e == 0],
                "Relapsed": scanb[scanb.RFi_e == 1]}

print("TCGA endo PFI no/yes:", len(tcga_groups["No relapse"]), len(tcga_groups["Relapsed"]),
      "| SCAN-B endo RFi no/yes:", len(scanb_groups["No relapse"]), len(scanb_groups["Relapsed"]))


# matrix + row scaling + stats

columns = [("TCGA", "No relapse"), ("TCGA", "Relapsed"),
           ("SCAN-B", "No relapse"), ("SCAN-B", "Relapsed")]

means = np.zeros((len(CELLS), 4))
for j, (cohort, group) in enumerate(columns):
    d = (tcga_groups if cohort == "TCGA" else scanb_groups)[group]
    means[:, j] = [d[c].mean() for c in CELLS]

# z-score each row within cohort so the two blocks are comparable
zscored = np.zeros_like(means)
for block in [slice(0, 2), slice(2, 4)]:
    sub = means[:, block]
    zscored[:, block] = (sub - sub.mean(1, keepdims=True)) / (sub.std(1, keepdims=True) + 1e-9)

# per-cohort Mann-Whitney, BH across the 22 cell types
pvals = {}
qvals = {}
for cohort, groups in [("TCGA", tcga_groups), ("SCAN-B", scanb_groups)]:
    ps = np.array([mannwhitney_p(groups["No relapse"][c], groups["Relapsed"][c])
                   for c in CELLS], dtype=float)
    pvals[cohort] = ps
    qvals[cohort] = bh_fdr(ps)

# heatmap

fig, ax = plt.subplots(figsize=(7.4, 9.2))
im = ax.imshow(zscored, cmap="RdBu_r",
               norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2), aspect="auto")

col_n = [len((tcga_groups if c == "TCGA" else scanb_groups)[g]) for c, g in columns]
ax.set_xticks(range(4))
ax.set_xticklabels(["%s\n(n=%d)" % (g, n) for (_, g), n in zip(columns, col_n)], fontsize=8)
ax.set_yticks(range(len(CELLS)))
ax.set_yticklabels(CELLS, fontsize=9)

# white gridlines between cells
ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(CELLS), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.4)
ax.tick_params(which="minor", length=0)

# annotate the relapsed column of each cohort with p / FDR
relapsed_col = {"TCGA": columns.index(("TCGA", "Relapsed")),
                "SCAN-B": columns.index(("SCAN-B", "Relapsed"))}

for cohort in ("TCGA", "SCAN-B"):
    j = relapsed_col[cohort]
    ps = pvals[cohort]
    qs = qvals[cohort]

    for i in range(len(CELLS)):
        sig = qs[i] == qs[i] and qs[i] < 0.05
        colour = "black" if sig else "#666666"
        weight = "bold" if sig else "normal"

        ax.text(j, i - 0.16, "p " + fmt_p(ps[i]), ha="center", va="center",
                fontsize=5.3, color=colour, fontweight=weight)
        ax.text(j, i + 0.20, "FDR " + fmt_p(qs[i]), ha="center", va="center",
                fontsize=5.3, color=colour, fontweight=weight)

        s = star(qs[i])
        if s:
            ax.text(j + 0.44, i - 0.30, s, ha="right", va="center",
                    fontsize=7, color="black")

ax.axvline(1.5, color="white", lw=3)
ax.text(0.5, -0.95, "TCGA (n=%d)" % len(tcga), ha="center", fontweight="bold", fontsize=9)
ax.text(2.5, -0.95, "SCAN-B (n=%d)" % len(scanb), ha="center", fontweight="bold", fontsize=9)

# outline the two polarised macrophage rows
for i, c in enumerate(CELLS):
    if c.startswith("Macrophages M1") or c.startswith("Macrophages M2"):
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 4, 1, fill=False,
                                   edgecolor="#e67e22", lw=1.8))

cb = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.02)
cb.set_label("Row-scaled\nimmune fraction", fontsize=8)

ax.set_title("Baseline immune profile by eventual outcome\n"
             "within ENDOCRINE-TREATED ER+ (relapsed vs not)  |  CIBERSORTx B-mode",
             fontsize=9.5, pad=30)

fig.text(0.5, 0.045,
         "Relapsed columns: p = Mann-Whitney U (relapsed vs no-relapse); "
         "FDR = Benjamini-Hochberg (22 cells). "
         "Bold = FDR<0.05. Stars from FDR. Row-scaled within cohort.",
         ha="center", fontsize=6.4, color="#555")

fig.subplots_adjust(bottom=0.12, top=0.9)
plt.savefig(OUT_PDF, bbox_inches="tight")
print("saved")
