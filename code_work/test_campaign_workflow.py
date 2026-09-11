#!/usr/bin/env python3
"""Integration test using THREE LEGACY GEOMETRIES, not new scientific results.

Recomputes thermal/search/ML outputs with tiny budgets, verifies checkpoint
reuse and tamper detection, then creates a clearly labelled smoke report.
Requires the three tracked legacy particle/contact dumps and Python dependencies.
Does not test LAMMPS generation. Takes --output for retaining a QA report.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from run_campaign30 import Campaign, CODE, SEEDS, write_json


def run(root):
    root.mkdir(parents=True,exist_ok=True)
    campaign = Campaign(root)
    settings = {"seeds":SEEDS[:3],"smoke_test":True,"random_samples":5,
                "search_restarts":2,"swap_proposals":10,"ml_repeats":1,
                "epochs":2,"thermal_model":"circular_constriction_4a_v2"}
    write_json(root/"campaign_manifest.json",{"settings":settings,
               "test_fixture":"Legacy geometry; Python integration test only, no new DEM"})
    py=lambda script:[sys.executable,"-u",str(CODE/script)]
    for seed in settings["seeds"]:
        case=f"packing_seed_{seed}"
        (root/case).mkdir(exist_ok=True)
        for filename in ["particles_final.dump","contacts_final.dump"]:
            shutil.copy2(CODE/case/filename,root/case/filename)
        write_json(root/"checkpoints"/f"dem_{seed}.json",
                   {"validation":{"kinetic_energy_J":0,"fixture":True},
                    "note":"No DEM was run; existing geometry tests Python execution only"})
        common=[case,"--k-low","1","--k-high","10","--k-fluid","0"]
        campaign.stage(f"baseline_{seed}",py("solve_packing_heat_transfer.py")+common+
                       ["--output","thermal_baseline"],[f"{case}/thermal_baseline"])
        campaign.stage(f"search_{seed}",py("run_allocation_pilot_stratified_v2.py")+
                       [case,"--output",f"{case}/allocation_final","--random-samples","5",
                        "--swap-proposals","10","--optimizer-restarts","2"],[f"{case}/allocation_final"])
        campaign.stage(f"verify_{seed}",py("solve_packing_heat_transfer.py")+common+
                       ["--output","thermal_optimized","--high-ids",f"{case}/allocation_final/high_ids_optimized.txt"],
                       [f"{case}/thermal_optimized"])
    campaign.stage("dataset",py("build_gnn_allocation_dataset.py")+
                   ["--root",".","--output","dataset","--allocation-template","{case}/allocation_final"],["dataset"])
    for seed in settings["seeds"]:
        case=f"packing_seed_{seed}"
        campaign.stage(f"ml_{seed}",py("train_evaluate_gnn_allocation_loocv.py")+
                       ["--dataset","dataset","--project-root",".","--output",f"ml/{case}",
                        "--test-case",case,"--device","cpu","--repeats","1","--epochs","2","--patience","2"],
                       [f"ml/{case}"])
        campaign.stage(f"analysis_{seed}",py("campaign_analysis.py")+["--root",".","--case",case],
                       [f"analysis/{case}"])
    campaign.stage("report",py("report_campaign30.py")+["--root","."],["report"])
    marker=root/"checkpoints/report.json"
    before=marker.read_bytes()
    campaign.stage("report",py("report_campaign30.py")+["--root","."],["report"])
    assert marker.read_bytes()==before, "Completed stage unexpectedly reran"
    pdf=root/"report/report.pdf"
    original=pdf.read_bytes()
    pdf.write_bytes(original+b"tampered")
    try:
        campaign.stage("report",py("report_campaign30.py")+["--root","."],["report"])
    except RuntimeError as exc:
        assert "Changed/missing" in str(exc)
    else:
        raise AssertionError("Tampered result was accepted")
    finally:
        pdf.write_bytes(original)
    print("PASS: Python integration, report creation, resume and tamper detection")


if __name__=="__main__":
    for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
        os.environ[name]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Choose an empty output directory")
        run(args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="heatpath_test_") as tmp:
            run(Path(tmp))
