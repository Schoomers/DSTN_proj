#!/usr/bin/env python3
"""
generate_fio_workloads.py

Generate a *suite* of fio job files (and a manifest CSV) covering common SSD workloads:
- sequential read/write
- random read/write
- mixed randrw with different read/write ratios via rwmixread
- hotspot random via random_distribution (zipf / pareto)
- mixed block sizes via bssplit
- varying iodepth and numjobs

References (fio docs):
- rw patterns (read/write/randread/randwrite/rw/randrw) and direct I/O: fio_doc.html (I/O type section)
- rwmixread / rwmixwrite for mixed workloads
- random_distribution for hotspot patterns (zipf/pareto/normal/zoned)
- bssplit for weighted block-size mixes
- ioengine, iodepth, numjobs, runtime/time_based

USAGE (typical):
  # 1) Put this file in: data_preparation/generate_fio_workloads.py
  # 2) Generate workloads:
  python3 data_preparation/generate_fio_workloads.py \
      --out_dir dataset/env/fio_workloads \
      --filename /mnt/nvme0n1 \
      --runtime 120 --time_based \
      --size 8G

Then run any generated job:
  fio dataset/env/fio_workloads/<jobname>.fio

NOTE:
- If you run against a real device, be careful. Many workloads write data.
- Use --readonly (fio option) or rw=read/randread if you want read-only tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    rw: str
    bs: str
    iodepth: int
    numjobs: int
    rwmixread: Optional[int] = None          # only for rw/randrw
    ioengine: str = "libaio"
    direct: int = 1
    runtime: int = 120
    time_based: int = 1
    size: str = "4G"
    random_distribution: Optional[str] = None  # e.g. zipf:1.2
    bssplit: Optional[str] = None              # e.g. 4k/80:64k/20
    norandommap: Optional[int] = None          # often used with non-uniform random dist
    group_reporting: int = 1
    ramp_time: Optional[int] = 5               # warm-up seconds, optional
    additional: Optional[Dict[str, Any]] = None


def _fmt_kv(k: str, v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        v = int(v)
    return f"{k}={v}\n"


def render_fio_job(spec: WorkloadSpec, filename: Optional[str], directory: Optional[str]) -> str:
    """
    Produce a single .fio file with a [global] section and one [job] section.
    """
    if (filename is None) and (directory is None):
        raise ValueError("Provide either --filename (block device/file) or --directory (dir for test files).")

    lines = []
    lines.append("[global]\n")

    # Target selection
    if filename is not None:
        lines.append(f"filename={filename}\n")
    else:
        # when using directory, fio creates files inside directory
        lines.append(f"directory={directory}\n")
        # Give each job its own file names
        lines.append("filename_format=fio.$jobnum.$filenum\n")

    # Common knobs
    lines.append(_fmt_kv("ioengine", spec.ioengine))
    lines.append(_fmt_kv("direct", spec.direct))
    lines.append(_fmt_kv("runtime", spec.runtime))
    lines.append(_fmt_kv("time_based", spec.time_based))
    lines.append(_fmt_kv("size", spec.size))
    lines.append(_fmt_kv("iodepth", spec.iodepth))
    lines.append(_fmt_kv("numjobs", spec.numjobs))
    lines.append(_fmt_kv("group_reporting", spec.group_reporting))

    if spec.ramp_time is not None and spec.ramp_time > 0:
        lines.append(_fmt_kv("ramp_time", spec.ramp_time))

    # I/O pattern + mix
    lines.append(_fmt_kv("rw", spec.rw))
    lines.append(_fmt_kv("bs", spec.bs))
    if spec.rwmixread is not None:
        lines.append(_fmt_kv("rwmixread", spec.rwmixread))

    # Optional hotspot / distributions
    if spec.random_distribution is not None:
        lines.append(_fmt_kv("random_distribution", spec.random_distribution))
    if spec.norandommap is not None:
        lines.append(_fmt_kv("norandommap", spec.norandommap))

    # Optional block-size weighting
    if spec.bssplit is not None:
        lines.append(_fmt_kv("bssplit", spec.bssplit))

    # Additional arbitrary key-values (advanced knobs)
    if spec.additional:
        for k, v in spec.additional.items():
            lines.append(_fmt_kv(k, v))

    # Single job section
    lines.append(f"\n[{spec.name}]\n")
    lines.append("stonewall\n")  # if users concatenate jobs, keep sequential boundaries

    return "".join([ln for ln in lines if ln])  # drop empty lines


def default_workloads(runtime: int, size: str, time_based: bool) -> List[WorkloadSpec]:
    """
    Reasonable coverage set. You can edit or extend this list.
    """
    tb = 1 if time_based else 0
    base = dict(runtime=runtime, size=size, time_based=tb)

    workloads: List[WorkloadSpec] = []

    # Classic baselines: seq + rand
    for bs in ["4k", "16k", "64k", "256k"]:
        workloads.append(WorkloadSpec(name=f"seqread_{bs}_qd1_nj1",  rw="read",     bs=bs, iodepth=1,  numjobs=1, **base))
        workloads.append(WorkloadSpec(name=f"seqwrite_{bs}_qd1_nj1", rw="write",    bs=bs, iodepth=1,  numjobs=1, **base))
        workloads.append(WorkloadSpec(name=f"randread_{bs}_qd32_nj4", rw="randread", bs=bs, iodepth=32, numjobs=4, **base))
        workloads.append(WorkloadSpec(name=f"randwrite_{bs}_qd32_nj4", rw="randwrite", bs=bs, iodepth=32, numjobs=4, **base))

    # Mixed R/W ratios via rwmixread (fio: rwmixread is "% reads" for mixed workloads)
    for mix in [95, 70, 50, 30, 5]:
        workloads.append(
            WorkloadSpec(
                name=f"randrw_4k_mix{mix}_qd32_nj4",
                rw="randrw",
                bs="4k",
                iodepth=32,
                numjobs=4,
                rwmixread=mix,
                **base
            )
        )

    # Hotspot reads: zipf distribution (skews random access)
    workloads.append(
        WorkloadSpec(
            name="randread_4k_zipf1p2_qd32_nj4",
            rw="randread",
            bs="4k",
            iodepth=32,
            numjobs=4,
            random_distribution="zipf:1.2",
            norandommap=1,
            **base,
        )
    )
    workloads.append(
        WorkloadSpec(
            name="randread_4k_pareto2p0_qd32_nj4",
            rw="randread",
            bs="4k",
            iodepth=32,
            numjobs=4,
            random_distribution="pareto:2.0",
            norandommap=1,
            **base,
        )
    )

    # Mixed block sizes using bssplit (weighted sizes)
    workloads.append(
        WorkloadSpec(
            name="randread_bssplit_4k80_64k20_qd32_nj4",
            rw="randread",
            bs="4k",
            iodepth=32,
            numjobs=4,
            bssplit="4k/80:64k/20",
            **base,
        )
    )
    workloads.append(
        WorkloadSpec(
            name="randrw_mix70_bssplit_4k70_64k30_qd32_nj4",
            rw="randrw",
            bs="4k",
            iodepth=32,
            numjobs=4,
            rwmixread=70,
            bssplit="4k/70:64k/30",
            **base,
        )
    )

    # Latency-sensitive (QD1) random
    for mix in [100, 70, 0]:
        rw = "randread" if mix == 100 else ("randwrite" if mix == 0 else "randrw")
        workloads.append(
            WorkloadSpec(
                name=f"latency_qd1_{rw}_4k_nj1" if mix in (0,100) else f"latency_qd1_randrw_4k_mix{mix}_nj1",
                rw=rw,
                bs="4k",
                iodepth=1,
                numjobs=1,
                rwmixread=(mix if rw == "randrw" else None),
                **base,
            )
        )

    return workloads


def write_manifest(manifest_path: Path, specs: List[WorkloadSpec], job_paths: List[Path]) -> None:
    """
    Save a CSV that maps workload specs to generated .fio files.
    """
    import csv

    fieldnames = sorted(set().union(*[asdict(s).keys() for s in specs])) + ["jobfile"]
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s, p in zip(specs, job_paths):
            row = asdict(s)
            row["jobfile"] = str(p)
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate fio job files and a manifest CSV.")
    ap.add_argument("--out_dir", required=True, help="Output directory for generated .fio jobs.")
    ap.add_argument("--filename", default=None, help="Target block device or file (e.g., /dev/nvme0n1).")
    ap.add_argument("--directory", default=None, help="Directory target (fio creates files in this dir).")
    ap.add_argument("--runtime", type=int, default=120, help="Seconds to run each workload.")
    ap.add_argument("--time_based", action="store_true", help="Loop workload for full runtime even if file is exhausted.")
    ap.add_argument("--size", default="4G", help="Size of region/files to operate within (e.g., 4G, 20%, etc.).")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be generated (no files).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if (args.filename is None) and (args.directory is None):
        raise SystemExit("ERROR: Provide --filename or --directory.")
    if (args.filename is not None) and (args.directory is not None):
        raise SystemExit("ERROR: Provide only one of --filename or --directory.")

    specs = default_workloads(runtime=args.runtime, size=args.size, time_based=args.time_based)

    if args.dry_run:
        print(f"Would generate {len(specs)} workloads into: {out_dir}")
        for s in specs[:10]:
            print("-", s.name)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    job_paths: List[Path] = []
    for spec in specs:
        job_text = render_fio_job(spec, filename=args.filename, directory=args.directory)
        job_path = out_dir / f"{spec.name}.fio"
        job_path.write_text(job_text, encoding="utf-8")
        job_paths.append(job_path)

    manifest_path = out_dir / "workloads_manifest.csv"
    write_manifest(manifest_path, specs, job_paths)

    print(f"Generated {len(job_paths)} fio job files in: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nExample run:")
    print(f"  fio {job_paths[0]}")


if __name__ == "__main__":
    main()


"""
python3 data_preparation/generate_fio_workloads.py \
  --out_dir dataset/env/fio_workloads \
  --filename /dev/nvme0n1 \
  --runtime 120 --time_based \
  --size 8G

  
  fio dataset/env/fio_workloads/randrw_4k_mix70_qd32_nj4.fio

"""