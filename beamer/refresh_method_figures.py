"""Regenerate method-labelled presentation figures from unchanged campaign tables."""
import csv
import io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parent
METHODS=["random","degree","heat_path","mlp","gnn","swap_optimized"]
LABELS=["Random\nallocation","Degree\nranking","Heat-throughput\nranking","MLP","GNN","Swap search"]
COLORS=["#8A8A8A","#D69F00","#009E73","#56B4E9","#0072B2","#D55E00"]
def read(name):
    with (ROOT/"data"/name).open() as f:return list(csv.DictReader(f))
rows=read("per_packing.csv")
values=np.array([[float(r[m]) for m in METHODS] for r in rows])
gains=np.array([[float(r[m+"_gain_percent"]) for m in METHODS] for r in rows])
n=len(rows)
mechanisms=read("mechanisms_all_packings.csv")
sensitivity=read("floor_sensitivity_all.csv")
def sd(x):return np.std(x,ddof=1)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"axes.unicode_minus":False})
def save(fig,name,caption):
    buf=io.BytesIO();fig.savefig(buf,format="pdf",bbox_inches="tight",facecolor="white")
    (ROOT/"figures"/(name+".pdf")).write_bytes(buf.getvalue());plt.close(fig)
fig,ax = plt.subplots(figsize=(7.1,4.1),layout="constrained")
for row in values:
    ax.plot(range(6),row,color="0.8",lw=0.5,zorder=1)
for j,color in enumerate(COLORS):
    ax.scatter(np.full(n,j),values[:,j],s=14,color=color,alpha=.7,zorder=2)
    ax.errorbar(j,values[:,j].mean(),yerr=sd(values[:,j]),fmt="_",ms=18,
                color="black",capsize=4,lw=1.2,zorder=3)
ax.set(xticks=range(6),xticklabels=LABELS,ylabel=r"Effective conductivity, $k_{\mathrm{eff}}$ [W m$^{-1}$ K$^{-1}$]")
ax.grid(axis="y",alpha=.18)
save(fig,"01_conductivity","Conductivity by allocation method. Each point is one packing; grey lines connect the same packing. Black marks show mean +/- sample SD. Random denotes each packing's random-allocation mean.")

fig,ax = plt.subplots(figsize=(7.1,4.1),layout="constrained")
for j in range(1,6):
    ax.scatter(np.full(n,j-1),gains[:,j],s=15,color=COLORS[j],alpha=.65)
    ax.errorbar(j-1,gains[:,j].mean(),yerr=sd(gains[:,j]),fmt="_",ms=18,color="black",capsize=4)
ax.axhline(0,color="0.4",lw=.8)
ax.set(xticks=range(5),xticklabels=LABELS[1:],ylabel="Improvement over random allocation [%]")
save(fig,"02_allocation_gain","Per-packing improvement relative to that packing's random mean; summary marks show mean +/- sample SD. This uses equal material-volume quotas in all methods.")

fig,axes=plt.subplots(1,2,figsize=(7.1,3.7),layout="constrained")
mm=["random_near_mean","heat_path","mlp","gnn","swap_optimized"]
for j,m in enumerate(mm):
    group=[r for r in mechanisms if r["method"] == m]
    for ax,key,scale in [(axes[0],"high_high_contact_heat_fraction",100),(axes[1],"spanning_high_cluster_count",1)]:
        y=np.array([float(r[key])*scale for r in group])
        ax.scatter(np.full(n,j),y,s=12,alpha=.6,color=[COLORS[i] for i in [0,2,3,4,5]][j])
        ax.plot(j,y.mean(),"_",ms=16,color="black")
for ax in axes:
    ax.set_xticks(range(5),["Random allocation","Heat-throughput\nranking","MLP","GNN","Swap search"],rotation=30,ha="right")
axes[0].set_ylabel("Heat on high-high contacts [%]")
axes[1].set_ylabel("Wall-spanning high-k clusters")
axes[1].set_ylim(-0.1,max(1.1,axes[1].get_ylim()[1]))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
save(fig,"04_mechanisms","Mechanisms across all packings. Random is the sampled allocation closest to the packing's random mean. High-high heat fraction uses absolute particle-contact heat rates. Black marks are ensemble means.")

fig,ax=plt.subplots(figsize=(7.1,4.1),layout="constrained")
for j,m in enumerate(mm):
    x=np.array([1e-10,1e-8,1e-6])
    y=np.array([[float(r["change_vs_nominal_percent"]) for r in sensitivity
                 if r["method"] == m and float(r["overlap_floor_ratio"]) == floor] for floor in x])
    ax.errorbar(x,y.mean(axis=1),yerr=y.std(axis=1,ddof=1),marker="o",lw=1,capsize=3,
                label=["Random allocation","Heat-throughput\nranking","MLP","GNN","Swap search"][j],
                color=[COLORS[i] for i in [0,2,3,4,5]][j])
ax.set(xscale="log",xlabel="Overlap floor / reference diameter",ylabel="Change in conductivity from nominal [%]")
ax.legend(frameon=False,ncol=3)
save(fig,"06_floor_sensitivity","Numerical overlap-floor sensitivity for fixed allocations: mean +/- sample SD over packings. Nominal floor is 1e-8. This is not a physical stiffness sensitivity or experimental validation.")

