env_paper_only_effects.csv
=========================

What this file contains
----------------------
This CSV is a *paper-only* ground-truth set of environmental-effect observations extracted from:
  - Temperature.pdf (temperature + humidity experiments)
  - Vibration.pdf (vibration experiments)

Each row is one quantitative claim explicitly stated in the papers (often in figure captions / result text),
expressed as: "under condition X, metric Y improves/degrades by Z% (or within a reported range)."

Important: This file intentionally DOES NOT try to reproduce/align the 'Experiment ID' tables in the paper.
It only records the numeric effect sizes that the paper states directly (e.g., “up to 75%”, “35–65%”, “67% at 60°C”).

Column meanings
---------------
- factor: temperature / humidity / temperature+humidity / vibration
- condition: human-readable description of the setting and the claim
- temperature_c: temperature in °C when explicitly stated; otherwise blank/NaN; may be 'decrease' for monotone-change claims
- humidity_pct: humidity info when explicitly stated; otherwise blank; may be 'increase', 'decrease', 'room', or 'high humidity (post-impact)'
- vibration_orientation: 'parallel', 'perpendicular', 'parallel vs perpendicular', or blank if not specified
- exposure: short-term / long-term / post-impact/long-term (paper wording)
- device_type: TLC / MLC / Vendor A/B/C / All (tested)
- workload: read / write / read+write / avg
- metric: 'tail_latency' or 'bandwidth'
- metric_percentile: e.g., 99th, 99.99th, average/mean, or 'tail (unspecified)' if the paper does not specify the percentile
- direction: 'improves' means better performance (lower latency or higher bandwidth); 'degrades' means worse performance
- change_pct_min, change_pct_max:
    * If the paper says “up to X%”, then min=0, max=X
    * If the paper gives a range “A% to B%”, then min=A, max=B
    * If the paper says “more than X%”, then min=X, max is blank
- source_paper: which PDF
- source_page: 1-indexed PDF page number where the statement appears
- source_fig: figure or 'text' label for easier tracing

How to use
----------
Copy this CSV into your repo at:
  dataset/env/env_paper_only_effects.csv

Then Stage II can treat each row as an evaluation prompt/ground-truth pair:
  Inputs  -> {factor + condition + (temperature_c/humidity_pct/orientation/exposure/device/workload)}
  Target  -> {metric + direction + change_pct_(min/max)}

Notes
-----
- Because these are “paper-only” statements, coverage is limited to what is explicitly written as numbers.
- Some claims are reported as 'tail latency' without specifying percentile; we kept them as such.
