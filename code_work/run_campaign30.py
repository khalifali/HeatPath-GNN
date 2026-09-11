#!/usr/bin/env python3
"""Generate and analyse 30 new packings. See ../docs/campaign30.md.

Examples:
  python3 run_campaign30.py --check --lammps lmp --mpi-ranks 8
  python3 run_campaign30.py --lammps lmp --mpi-ranks 8
  python3 run_campaign30.py --dry-run
Repeat the same command to resume. Completed stages are content-verified.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import solve_packing_heat_transfer as thermal

CODE = Path(__file__).resolve().parent
SEEDS = [18427, 29173, 37811, 46549, 55291, 64109, 72869, 81647, 90439, 99223,
         108301, 117307, 126323, 135347, 144359, 153371, 162389, 171401,
         180413, 189431, 198439, 207457, 216481, 225503, 234529, 243551,
         252563, 261581, 270593, 279607]
SOURCES = ["run_campaign30.py", "report_campaign30.py", "campaign_analysis.py",
           "in.generate_equal_porosity_psd.lammps", "solve_packing_heat_transfer.py",
           "run_allocation_pilot_stratified_v2.py", "build_gnn_allocation_dataset.py",
           "train_evaluate_gnn_allocation_loocv.py", "analyze_allocation_mechanisms.py",
           "test_contact_model.py", "test_heat_transfer_solver.py", "test_lammps_contact.py"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


# Only this audited predecessor may retain its completed stages on upgrade.
RELAX_PREDECESSOR = {'run_campaign30.py': 'c0daf8cdee31a41f394c7bdf4ffbbcfa12ef3dda0349ada5aa04bf5cb3021f14', 'in.generate_equal_porosity_psd.lammps': '1ece1f27d1bc5d1e97283ecd3169c0b7a47db3e0996c65f5876f889b45395423'}


def upgrade_relaxation(root, previous, current):
    expected = json.loads(json.dumps(current))
    expected["source_sha256"].update(RELAX_PREDECESSOR)
    expected["settings"].pop("relaxation_extension", None)
    if previous != expected:
        raise RuntimeError("This campaign is not the supported growth-fix predecessor, "
                           "or its settings/environment changed; cannot upgrade in place")
    # Verify all completed products before accepting the migration.
    for marker in sorted((root/"checkpoints").glob("*.json")):
        saved = json.loads(marker.read_text())
        for path, sha in saved["outputs"].items():
            if not (root/path).is_file() or digest(root/path) != sha:
                raise RuntimeError(f"Cannot upgrade: changed/missing completed output {path}")
    archive = root/"provenance_before_relaxation_extension"
    archive.mkdir(exist_ok=True)
    archived_manifest = archive/"campaign_manifest.json"
    if archived_manifest.exists() and json.loads(archived_manifest.read_text()) != previous:
        raise RuntimeError("Conflicting archived campaign manifest")
    for name, sha in previous["source_sha256"].items():
        original = root/"source_snapshot"/name
        target = archive/name
        if target.exists():
            if digest(target) != sha:
                raise RuntimeError(f"Conflicting archived source {name}")
        else:
            if not original.is_file() or digest(original) != sha:
                raise RuntimeError(f"Cannot upgrade: original source snapshot changed: {name}")
            shutil.copy2(original, target)
    write_json(archived_manifest, previous)
    write_json(archive/"upgrade.json", {
        "reason": "Extend unconverged packings after the unchanged 200000-step minimum",
        "retained_checkpoints": [p.name for p in sorted((root/"checkpoints").glob("*.json"))],
        "new_signature": current})
    write_json(root/"campaign_manifest.json", current)
    print("[upgrade] Preserved verified completed stages and archived original provenance", flush=True)


def validate_packing(case, max_ke):
    particles, box = thermal.read_particles(case / "particles_final.dump")
    contacts = thermal.read_contact_ids(case / "contacts_final.dump")
    ids = [p.atom_id for p in particles]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError(f"{case.name}: expected 500 unique particles")
    for typ, count, diameter in [(1,100,0.0015),(2,300,0.002),(3,100,0.0025)]:
        group = [p for p in particles if p.atom_type == typ]
        if len(group) != count or not all(math.isclose(2*p.radius,diameter,rel_tol=1e-10) for p in group):
            raise ValueError(f"{case.name}: wrong PSD")
    volume = sum(4*math.pi*p.radius**3/3 for p in particles)
    porosity = 1-volume/np.prod(box.lengths)
    if abs(porosity-0.38) > 1e-7:
        raise ValueError(f"{case.name}: porosity {porosity} differs from 0.38")
    with (case / "packing_summary.dat").open() as stream:
        rows = list(csv.DictReader(stream, delimiter=" ", skipinitialspace=True))
    summary = rows[-1]
    ke = float(summary["kineticEnergy_J"])
    if not math.isfinite(ke) or ke < 0 or ke > max_ke:
        raise ValueError(f"{case.name}: final KE={ke:g} J exceeds {max_ke:g}; inspect DEM log")
    pairs = {tuple(sorted(pair)) for pair in contacts}
    if len(pairs) != len(contacts) or any(a == b or a not in ids or b not in ids for a,b in pairs):
        raise ValueError(f"{case.name}: duplicate, self or invalid contacts")
    # Node and contact snapshots must describe the same DEM timestep.
    steps = [int((case / f).read_text().splitlines()[1]) for f in
             ("particles_final.dump", "contacts_final.dump")]
    if steps[0] != steps[1]:
        raise ValueError(f"{case.name}: particle/contact snapshot mismatch")
    edges,_ = thermal.build_contact_edges(particles,box,contacts,np.ones(500),2e-11)
    hot,cold,_,_ = thermal.build_wall_links(particles,box,np.ones(500),math.inf,2e-11,2e-11,0,0,0)
    diagnostics = thermal.graph_diagnostics(500,edges,hot,cold)
    if diagnostics["spanning_component_count"] < 1:
        raise ValueError(f"{case.name}: no wall-spanning contact component")
    xyz = np.array([[p.x,p.y,p.z] for p in particles])
    index = {p.atom_id:i for i,p in enumerate(particles)}
    gaps = [np.linalg.norm(thermal.minimum_image_vector(xyz[index[a]],xyz[index[b]],box))
            -particles[index[a]].radius-particles[index[b]].radius for a,b in pairs]
    if max(gaps) > 2e-10:
        raise ValueError(f"{case.name}: contact list includes separated particles")
    return {"case":case.name,"porosity":float(porosity),"kinetic_energy_J":ke,
            "contacts":len(edges),"snapshot_step":steps[0],"graph":diagnostics}


class Campaign:
    def __init__(self, root):
        self.root = root
        self.steps = root / "checkpoints"
        self.steps.mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)

    def stage(self, name, command, outputs, validator=None):
        marker = self.steps / f"{name}.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["command"] != command:
                raise RuntimeError(f"Changed command for completed stage {name}; use a new output directory")
            for path, sha in saved["outputs"].items():
                if not (self.root/path).is_file() or digest(self.root/path) != sha:
                    raise RuntimeError(f"Changed/missing completed output {path}; restore it or use a new campaign")
            print(f"[skip] {name}", flush=True)
            return
        print(f"[run] {name}  (log: logs/{name}.log)", flush=True)
        started = time.monotonic()
        log = self.root / "logs" / f"{name}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=self.root, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
            raise RuntimeError(f"{name} failed (exit {result.returncode})\n{tail}\nFull log: {log}")
        evidence = validator() if validator else None
        files = []
        for path in outputs:
            path = self.root / path
            if not path.exists():
                raise RuntimeError(f"{name}: missing output {path}")
            files.extend(sorted(path.rglob("*")) if path.is_dir() else [path])
        hashes = {str(p.relative_to(self.root)):digest(p) for p in files if p.is_file()}
        write_json(marker,{"command":command,"elapsed_s":time.monotonic()-started,
                           "outputs":hashes,"validation":evidence})
        print(f"[done] {name}",flush=True)


def preflight(args):
    import torch
    import matplotlib
    import reportlab
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    executable = shutil.which(args.lammps)
    if not executable:
        raise RuntimeError(f"LAMMPS executable not found: {args.lammps}")
    help_run = subprocess.run([executable,"-h"],capture_output=True,text=True,check=True)
    if "GRANULAR" not in help_run.stdout:
        raise RuntimeError("LAMMPS must be built with the GRANULAR package")
    if args.mpi_ranks > 1 and not shutil.which(args.mpi):
        raise RuntimeError(f"MPI launcher not found: {args.mpi}")
    for test in ("test_contact_model.py","test_heat_transfer_solver.py"):
        subprocess.run([sys.executable,str(CODE/test)],check=True,cwd=CODE)
    subprocess.run([sys.executable,str(CODE/"test_lammps_contact.py"),"--lammps",executable],check=True)
    return executable, help_run.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=CODE.parent/"runs"/"campaign30")
    parser.add_argument("--lammps",default="lmp",help="LAMMPS executable name or absolute path")
    parser.add_argument("--mpi",default="mpirun",help="MPI launcher, compiled for this LAMMPS installation")
    parser.add_argument("--mpi-ranks",type=int,default=8)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--max-ke",type=float,default=1e-12,help="Maximum final translational KE [J]")
    parser.add_argument("--check",action="store_true",help="Check dependencies and analytical tests, then exit")
    parser.add_argument("--dry-run",action="store_true",help="Show plan without launching or writing")
    parser.add_argument("--smoke-test",action="store_true",help="3 full DEM packings, tiny search/ML budgets; NOT publication results")
    parser.add_argument("--upgrade-relaxation", action="store_true",
                        help="Migrate the verified growth-fix campaign; preserve completed stages")
    args = parser.parse_args()
    if args.mpi_ranks < 1 or args.max_ke <= 0:
        parser.error("MPI ranks and KE threshold must be positive")
    seeds = SEEDS[:3] if args.smoke_test else SEEDS
    settings = {"seeds":seeds,"smoke_test":args.smoke_test,"mpi_ranks":args.mpi_ranks,
                "device":args.device,"max_ke_J":args.max_ke,
                "random_samples":5 if args.smoke_test else 100,
                "swap_proposals":10 if args.smoke_test else 3000,
                "search_restarts":2 if args.smoke_test else 5,
                "ml_repeats":1 if args.smoke_test else 3,
                "epochs":2 if args.smoke_test else 400,"patience":2 if args.smoke_test else 60,
                "thermal_model":"circular_constriction_4a_v2",
                "E_Pa":5e7,"poisson":0.25,"restitution":0.30,"friction":0.50,
                "k_low":1.0,"k_high":10.0,"type_quotas":{"1":10,"2":30,"3":10},
                "relaxation_extension":{"block_steps":50000,"max_total_steps":2000000,"target_ke_J":1e-12},
                "growth_steps":300000,"relaxation_steps":200000,"dt_s":2e-7}
    root = args.output.expanduser().resolve()
    if args.smoke_test and root == (CODE.parent/"runs"/"campaign30").resolve():
        root = root.with_name("smoke_test")
    if args.dry_run:
        print(json.dumps({"output":str(root),"settings":settings,
                         "sequence":["DEM + QA per seed","baseline + search + verification",
                                     "dataset","one ML stage per held-out packing",
                                     "mechanisms + sensitivity per packing","report and figures"]},indent=2))
        return
    executable, lammps_help = preflight(args)
    if args.check:
        print("Preflight passed: dependencies, Python models and a static LAMMPS contact. Full packing generation is a separate stage.")
        return
    if root.exists() and any(root.iterdir()) and not (root/"campaign_manifest.json").exists():
        raise RuntimeError("Output directory is not an existing campaign or empty; choose a new directory")
    root.mkdir(parents=True,exist_ok=True)
    with (root/"campaign.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is running this campaign")
        signature = {"settings":settings,"source_sha256":{p:digest(CODE/p) for p in SOURCES},
                     "python":platform.python_version(),
                     "packages":{p:importlib.metadata.version(p) for p in
                                 ["numpy","scipy","torch","matplotlib","reportlab"]},
                     "lammps_executable":executable,"lammps_sha256":digest(executable),
                     "lammps_help_sha256":hashlib.sha256(lammps_help.encode()).hexdigest()}
        manifest = root/"campaign_manifest.json"
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if previous != signature:
                if args.upgrade_relaxation:
                    upgrade_relaxation(root, previous, signature)
                else:
                    raise RuntimeError("Code, settings or environment changed. For the growth-fix "
                                       "campaign, pass --upgrade-relaxation; otherwise use a new output directory.")
        else:
            write_json(manifest,signature)
        source = root/"source_snapshot"
        source.mkdir(exist_ok=True)
        for p in SOURCES:
            shutil.copy2(CODE/p,source/p)
        (root/"lammps_help.txt").write_text(lammps_help)
        campaign = Campaign(root)
        py = lambda name: [sys.executable,"-u",str(CODE/name)]
        for seed in seeds:
            case = f"packing_seed_{seed}"
            dem = ([args.mpi,"-np",str(args.mpi_ranks)] if args.mpi_ranks > 1 else [])
            dem += [executable,"-log",f"logs/dem_{seed}_lammps.log","-var","seed_realization",str(seed),
                    "-in",str(CODE/"in.generate_equal_porosity_psd.lammps")]
            campaign.stage(f"dem_{seed}",dem,[f"{case}/{f}" for f in
                           ["particles_final.dump","contacts_final.dump","packing_summary.dat",
                            "packing_final.data","packing_final.restart"]],
                           lambda c=root/case:validate_packing(c,args.max_ke))
            common = [case,"--k-low","1","--k-high","10","--k-fluid","0","--t-hot","301","--t-cold","300"]
            campaign.stage(f"baseline_{seed}",py("solve_packing_heat_transfer.py")+common+
                           ["--output","thermal_baseline"],[f"{case}/thermal_baseline"])
            allocation = f"{case}/allocation_final"
            campaign.stage(f"search_{seed}",py("run_allocation_pilot_stratified_v2.py")+
                           [case,"--output",allocation,"--random-samples",str(settings["random_samples"]),
                            "--swap-proposals",str(settings["swap_proposals"]),"--optimizer-restarts",str(settings["search_restarts"]),
                            "--type-quotas","1:10,2:30,3:10","--seed","12345"],[allocation])
            campaign.stage(f"verify_{seed}",py("solve_packing_heat_transfer.py")+common+
                           ["--output","thermal_optimized","--high-ids",f"{allocation}/high_ids_optimized.txt"],
                           [f"{case}/thermal_optimized"])
        campaign.stage("dataset",py("build_gnn_allocation_dataset.py")+
                       ["--root",".","--output","dataset","--allocation-template","{case}/allocation_final"],
                       ["dataset"])
        for seed in seeds:
            case = f"packing_seed_{seed}"
            campaign.stage(f"ml_{seed}",py("train_evaluate_gnn_allocation_loocv.py")+
                           ["--dataset","dataset","--project-root",".","--output",f"ml/{case}",
                            "--test-case",case,"--device",args.device,"--repeats",str(settings["ml_repeats"]),
                            "--epochs",str(settings["epochs"]),"--patience",str(settings["patience"]),
                            "--seed","20260911"],[f"ml/{case}"])
            campaign.stage(f"analysis_{seed}",py("campaign_analysis.py")+["--root",".","--case",case],
                           [f"analysis/{case}"])
        campaign.stage("report",py("report_campaign30.py")+["--root","."],["report"])
        print(f"Complete: {root/'report'/'report.pdf'}",flush=True)


if __name__ == "__main__":
    # One computational job at a time; avoid BLAS/PyTorch oversubscription.
    for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
        os.environ[name] = "1"
    try:
        main()
    except (RuntimeError,ValueError,subprocess.CalledProcessError,ImportError) as exc:
        sys.exit(str(exc))
