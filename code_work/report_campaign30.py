#!/usr/bin/env python3
"""Create reproducible tables, vector figures and a PDF/HTML/Markdown report."""
import argparse
import csv
import html
import io
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib import font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

METHODS = ["random", "degree", "heat_path", "mlp", "gnn", "swap_optimized"]
LABELS = ["Random", "Degree", "Heat path", "MLP", "GNN", "Swap search"]
COLORS = ["#8A8A8A", "#D69F00", "#009E73", "#56B4E9", "#0072B2", "#D55E00"]


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sd(values):
    return float(np.std(values,ddof=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/"campaign_manifest.json").read_text())
    config = manifest["settings"]
    seeds = config["seeds"]
    n = len(seeds)
    report = root/"report"
    figures = report/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    rows, ml_rows, mechanisms, contacts, sensitivity, checks = [],[],[],[],[],[]
    for seed in seeds:
        case = f"packing_seed_{seed}"
        alloc = json.loads((root/case/"allocation_final/allocation_summary.json").read_text())
        baseline = json.loads((root/case/"thermal_baseline/thermal_summary.json").read_text())
        verified = json.loads((root/case/"thermal_optimized/thermal_summary.json").read_text())
        if alloc.get("thermal_model") != config["thermal_model"]:
            raise ValueError("Mixed model provenance")
        if abs(verified["effective_conductivity_W_mK"]-alloc["optimized_best_k_eff_W_mK"]) > 1e-9:
            raise ValueError("Independent verification mismatch")
        fold = read_csv(root/"ml"/case/"loocv_results.csv")
        ml_rows.extend(fold)
        ensemble = {r["model"]:r for r in fold if r["repeat"] == "ensemble"}
        if set(ensemble) != {"mlp","gnn"}:
            raise ValueError(f"Missing model ensemble in {case}")
        row = {"seed":seed,"homogeneous":baseline["effective_conductivity_W_mK"],
               "random":alloc["random_mean_k_eff_W_mK"],"random_sample_sd":alloc["random_std_k_eff_W_mK"],
               "degree":alloc["degree_k_eff_W_mK"],"heat_path":alloc["heat_path_k_eff_W_mK"],
               "mlp":float(ensemble["mlp"]["effective_conductivity_W_mK"]),
               "gnn":float(ensemble["gnn"]["effective_conductivity_W_mK"]),
               "swap_optimized":alloc["optimized_best_k_eff_W_mK"],
               "search_restart_mean":alloc["optimized_mean_k_eff_W_mK"],
               "teacher_actual_solves":alloc["actual_thermal_solves"],
               "teacher_unique_allocations":alloc["unique_thermal_evaluations"],
               "max_energy_imbalance":max(alloc["maximum_energy_balance_error"],
                                         verified["relative_energy_balance_error"])}
        if row["max_energy_imbalance"] > 1e-8:
            raise ValueError("Energy-balance gate failed")
        if row["swap_optimized"] <= row["random"]:
            raise ValueError("Search gain is nonpositive; recovery fraction is undefined")
        rows.append(row)
        mechanisms.extend(read_csv(root/"analysis"/case/"mechanisms.csv"))
        contacts.extend(read_csv(root/"analysis"/case/"contact_geometry.csv"))
        sensitivity.extend(read_csv(root/"analysis"/case/"floor_sensitivity.csv"))
        checks.append(json.loads((root/"checkpoints"/f"dem_{seed}.json").read_text())["validation"])
    values = np.array([[r[m] for m in METHODS] for r in rows])
    gains = 100*(values/values[:,[0]]-1)
    recovery = 100*(values-values[:,[0]])/(values[:,[-1]]-values[:,[0]])
    summaries = []
    for j,method in enumerate(METHODS):
        summaries.append({"method":method,"packings":n,"mean_k_W_mK":float(values[:,j].mean()),
                          "sd_k_W_mK":sd(values[:,j]),"mean_gain_percent":float(gains[:,j].mean()),
                          "sd_gain_percent":sd(gains[:,j]),"mean_recovery_percent":float(recovery[:,j].mean()),
                          "sd_recovery_percent":sd(recovery[:,j])})
    for i,row in enumerate(rows):
        for j,m in enumerate(METHODS):
            row[m+"_gain_percent"] = gains[i,j]
            row[m+"_recovery_percent"] = recovery[i,j]
    write_csv(report/"per_packing.csv",rows)
    write_csv(report/"method_summary.csv",summaries)
    write_csv(report/"ml_all_repeats.csv",ml_rows)
    write_csv(report/"mechanisms_all_packings.csv",mechanisms)
    write_csv(report/"contact_geometry_all.csv",contacts)
    write_csv(report/"floor_sensitivity_all.csv",sensitivity)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
                         "savefig.dpi":600,"axes.unicode_minus":False})
    captions = []
    smoke = config["smoke_test"]

    def save(fig,name,caption):
        if smoke:
            fig.suptitle("SMOKE TEST - NOT SCIENTIFIC RESULTS",fontsize=10,color="#B22222")
        for ext in ["pdf","svg","png"]:
            # Complete the encoding in memory before writing the output file.
            buffer = io.BytesIO()
            fig.savefig(buffer,format=ext,bbox_inches="tight",facecolor="white")
            (figures/f"{name}.{ext}").write_bytes(buffer.getvalue())
        plt.close(fig)
        captions.append((name,caption))

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

    fig,ax = plt.subplots(figsize=(7.1,4.1),layout="constrained")
    x=np.arange(1,n+1)
    ax.plot(x,recovery[:,3],"o-",ms=4,lw=1,color=COLORS[3],label="MLP")
    ax.plot(x,recovery[:,4],"s-",ms=4,lw=1,color=COLORS[4],label="GNN")
    ax.axhline(100,color="0.4",ls="--",lw=.8,label="Best recorded search")
    ax.set(xlabel="Packing index (seed mapping in per_packing.csv)",ylabel="Recovered search gain [%]",xticks=x[::max(1,n//10)])
    ax.legend(frameon=False,ncol=3,loc="best")
    save(fig,"03_ml_recovery","Held-out-packing results. Recovery = 100 (k_model - k_random)/(k_best_search - k_random). The reference is the best fixed-budget search, not a proven global optimum. Models use identical node features and packing-level splits.")

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.7),layout="constrained")
    mm=["random_near_mean","heat_path","mlp","gnn","swap_optimized"]
    for j,m in enumerate(mm):
        group=[r for r in mechanisms if r["method"] == m]
        for ax,key,scale in [(axes[0],"high_high_contact_heat_fraction",100),(axes[1],"spanning_high_cluster_count",1)]:
            y=np.array([float(r[key])*scale for r in group])
            ax.scatter(np.full(n,j),y,s=12,alpha=.6,color=[COLORS[i] for i in [0,2,3,4,5]][j])
            ax.plot(j,y.mean(),"_",ms=16,color="black")
    for ax in axes:
        ax.set_xticks(range(5),["Random","Heat path","MLP","GNN","Search"],rotation=30,ha="right")
    axes[0].set_ylabel("Heat on high-high contacts [%]")
    axes[1].set_ylabel("Wall-spanning high-k clusters")
    axes[1].set_ylim(-0.1,max(1.1,axes[1].get_ylim()[1]))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig,"04_mechanisms","Mechanisms across all packings. Random is the sampled allocation closest to the packing's random mean. High-high heat fraction uses absolute particle-contact heat rates. Black marks are ensemble means.")

    fig,ax=plt.subplots(figsize=(7.1,4.1),layout="constrained")
    contact_ratios=np.array([float(r["a_over_min_radius"]) for r in contacts])
    ax.hist(contact_ratios,bins=35,color=COLORS[4],edgecolor="white",linewidth=.5)
    ax.set(xlabel=r"Contact radius / smaller particle radius, $a/\min(R_i,R_j)$",ylabel="Number of contacts")
    save(fig,"05_contact_geometry","Pooled contact-size distribution. Contacts within a packing are correlated; this histogram is a geometry diagnostic, not an independent-sample statistical test. Large ratios warrant checking the small-contact approximation.")

    fig,ax=plt.subplots(figsize=(7.1,4.1),layout="constrained")
    for j,m in enumerate(mm):
        x=np.array([1e-10,1e-8,1e-6])
        y=np.array([[float(r["change_vs_nominal_percent"]) for r in sensitivity
                     if r["method"] == m and float(r["overlap_floor_ratio"]) == floor] for floor in x])
        ax.errorbar(x,y.mean(axis=1),yerr=y.std(axis=1,ddof=1),marker="o",lw=1,capsize=3,
                    label=["Random","Heat path","MLP","GNN","Search"][j],
                    color=[COLORS[i] for i in [0,2,3,4,5]][j])
    ax.set(xscale="log",xlabel="Overlap floor / reference diameter",ylabel="Change in conductivity from nominal [%]")
    ax.legend(frameon=False,ncol=3)
    save(fig,"06_floor_sensitivity","Numerical overlap-floor sensitivity for fixed allocations: mean +/- sample SD over packings. Nominal floor is 1e-8. This is not a physical stiffness sensitivity or experimental validation.")

    gnn_wins=int(np.count_nonzero(values[:,4]>values[:,3]))
    max_error=max(r["max_energy_imbalance"] for r in rows)
    max_sensitivity=max(abs(float(r["change_vs_nominal_percent"])) for r in sensitivity)
    inference={m:[float(r["inference_time_s"]) for r in ml_rows if r["repeat"]=="ensemble" and r["model"]==m] for m in ["mlp","gnn"]}
    settings_text=(f"{n} independent 500-particle packings; diameters 1.5/2.0/2.5 mm with counts 100/300/100; "
                   "nominal porosity 0.38. E = 50 MPa (softened packing modulus), Poisson ratio 0.25, "
                   "restitution 0.30 and friction 0.50; no gravity, periodic x/y, fixed z walls. "
                   "Contact-only thermal conduction, k_low = 1 and k_high = 10 W/(m K), "
                   "wall temperatures 301/300 K and fixed 10/30/10 allocation quotas.")
    protocol=(f"Each packing uses {config['random_samples']} random allocations and {config['search_restarts']} "
              f"search restarts of {config['swap_proposals']} proposed swaps. Leave-one-packing-out evaluation uses "
              f"{n-2} training packings, one validation packing and one test packing. Each model has "
              f"{config['ml_repeats']} repeats, combined by within-type ranks. Maximum epochs: {config['epochs']}. "
              "Weights, training-only scalers and split identities are saved in each fold's models directory.")
    finding=(f"GNN exceeds MLP on {gnn_wins}/{n} held-out packings. Mean GNN recovery is "
             f"{recovery[:,4].mean():.1f}% (SD {sd(recovery[:,4]):.1f} percentage points). "
             f"Mean best-search improvement is {gains[:,5].mean():.1f}% over random. "
             f"Maximum recorded search/verification energy imbalance is {max_error:.3g}. "
             f"Maximum absolute floor-sensitivity change is {max_sensitivity:.3g}%.")
    geometry=(f"Contact a/min(R): median {np.median(contact_ratios):.3f}, 95th percentile "
              f"{np.quantile(contact_ratios,.95):.3f}, maximum {max(contact_ratios):.3f}. "
              f"Final translational KE: maximum {max(c['kinetic_energy_J'] for c in checks):.3g} J. "
              "All packings passed count, PSD, porosity, snapshot and wall-spanning connectivity checks.")
    if manifest.get("test_fixture"):
        geometry = "Legacy geometries were used only to test Python execution. No new DEM generation or packing-generation QA is claimed for this smoke report."
    cost=(f"Mean actual thermal solves during teacher search: {np.mean([r['teacher_actual_solves'] for r in rows]):.1f}. "
          f"Mean ensemble scoring time: MLP {np.mean(inference['mlp'])*1000:.3f} ms, "
          f"GNN {np.mean(inference['gnn'])*1000:.3f} ms. These scoring timings exclude the baseline and "
          "verification solves. Stage wall times are in checkpoints/*.json. No total wall-time speedup is claimed.")
    limitation=("Results apply to stochastic realizations of this packing specification. The contact model assumes "
                "ideal constriction resistance; softened mechanics and relatively large contacts require physical "
                "sensitivity assessment. Convection, radiation and fluid-gap conduction are excluded. "
                "Search results are best-found assignments. Reported SD describes packing variability; "
                "overlapping cross-validation training sets mean fold results are not independent training experiments. "
                "This campaign is not an untouched external test set. No R-squared metric is used for allocation quality.")
    title=f"HeatPath-GNN: {n}-packing campaign"
    if smoke:
        title="SMOKE TEST - NOT SCIENTIFIC RESULTS"
    sections=[("Configuration",settings_text),("Evaluation protocol",protocol),("Results",finding),
              ("Geometry and verification",geometry),("Computational work",cost),("Interpretation limits",limitation)]
    table_header=["Method","Mean k [W/(m K)]","SD k","Gain [%]","Recovery [%]"]
    table_rows=[[LABELS[i],f"{r['mean_k_W_mK']:.5f}",f"{r['sd_k_W_mK']:.5f}",
                 f"{r['mean_gain_percent']:.1f}",f"{r['mean_recovery_percent']:.1f}"] for i,r in enumerate(summaries)]
    md=f"# {title}\n\n"+"\n\n".join(f"## {h}\n\n{t}" for h,t in sections)
    md+="\n\n| "+" | ".join(table_header)+" |\n|"+"---|"*5+"\n"
    md+="\n".join("| "+" | ".join(row)+" |" for row in table_rows)
    for name,caption in captions:
        md+=f"\n\n![{name}](figures/{name}.png)\n\n{caption}"
    md+="\n\nSources: [LAMMPS granular model](https://docs.lammps.org/pair_granular.html). "
    md+="Exact code and package provenance: ../campaign_manifest.json and ../source_snapshot/.\n"
    (report/"report.md").write_text(md)
    body=f"<h1>{html.escape(title)}</h1>"+"".join(f"<h2>{h}</h2><p>{html.escape(t)}</p>" for h,t in sections)
    body+="<table><tr>"+"".join(f"<th>{h}</th>" for h in table_header)+"</tr>"
    body+="".join("<tr>"+"".join(f"<td>{v}</td>" for v in row)+"</tr>" for row in table_rows)+"</table>"
    body+="".join(f'<figure><img src="figures/{name}.svg"><figcaption>{html.escape(cap)}</figcaption></figure>' for name,cap in captions)
    (report/"report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(title)+
        '</title><style>body{max-width:950px;margin:40px auto;font:16px/1.6 sans-serif;color:#172b3a}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}figure{margin:40px 0}figcaption{font-size:14px}</style>'+body+'</html>')
    pdfmetrics.registerFont(TTFont("ReportSans",font_manager.findfont("DejaVu Sans")))
    pdfmetrics.registerFont(TTFont("ReportSansBold",font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans",weight="bold"))))
    styles=getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "ReportSansBold" if style.name.startswith("Heading") or style.name == "Title" else "ReportSans"
    story=[Paragraph(title,styles["Title"]),Spacer(1,.4*cm)]
    for h,t in sections[:3]:
        story.extend([Paragraph(h,styles["Heading2"]),Paragraph(t,styles["BodyText"])])
    table=Table([table_header]+table_rows,colWidths=[3.1*cm,4.2*cm,2.4*cm,2.5*cm,3.3*cm],repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E6EEF3")),
                              ("FONTNAME",(0,0),(-1,-1),"ReportSans"),("FONTSIZE",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),7),
                              ("TOPPADDING",(0,0),(-1,-1),7),("LINEBELOW",(0,0),(-1,0),.7,colors.grey)]))
    story.extend([Spacer(1,.4*cm),table,PageBreak()])
    for h,t in sections[3:]:
        story.extend([Paragraph(h,styles["Heading2"]),Paragraph(t,styles["BodyText"]),Spacer(1,.3*cm)])
    story.append(Paragraph("Model reference: https://docs.lammps.org/pair_granular.html. "
                           "Exact source hashes and environment are stored in campaign_manifest.json.",styles["BodyText"]))
    for i,(name,caption) in enumerate(captions,1):
        story.extend([PageBreak(),Paragraph(f"Figure {i}",styles["Heading1"])])
        img=Image(str(figures/f"{name}.png"))
        width=17*cm
        img.drawHeight=img.imageHeight*width/img.imageWidth
        img.drawWidth=width
        story.extend([img,Spacer(1,.4*cm),Paragraph(caption,styles["BodyText"])])
    def footer(canvas,doc):
        canvas.setFont("ReportSans",8)
        canvas.drawRightString(A4[0]-1.8*cm,1.1*cm,f"HeatPath-GNN | {doc.page}")
    SimpleDocTemplate(str(report/"report.pdf"),pagesize=A4,rightMargin=1.8*cm,leftMargin=1.8*cm,
                      topMargin=1.7*cm,bottomMargin=1.7*cm).build(story,onFirstPage=footer,onLaterPages=footer)
    (report/"figure_captions.md").write_text("\n\n".join(f"**{name}**: {caption}" for name,caption in captions))
    print(f"Wrote {report/'report.pdf'} and six PDF/SVG/600-dpi PNG figures")


if __name__ == "__main__":
    main()
