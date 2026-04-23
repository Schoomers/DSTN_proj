#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fleet-level prompt templates (Table II / cohort evaluation).

A fleet-level query analyzes a cohort of N drives at once. Unlike the per-window
(single-sample) prompts, predictive output MUST enumerate the subset of drives
predicted to fail so we can compute precision/recall/accuracy at the drive level.
"""

from __future__ import annotations
from typing import Any, Dict

def fleet_predictive_user_prompt(payload: Dict[str, Any]) -> str:
    return f"""Task: Fleet-level predictive analysis (cohort of drives).
Input JSON (payload): {payload}

Return EXACTLY ONE JSON object with this schema:
{{
  "task": "predictive",
  "cohort_id": <string>,
  "cohort_size": <integer>,
  "predicted_failing_drives": [
    {{
      "drive_id": <string>,
      "predicted_failure": 1,
      "predicted_ttf_days": <number|null>,
      "risk_factors": [<text>, ...],
      "support": [<ref_id>, ...]
    }}
  ],
  "rationale": <short text>,
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Only list drives you believe are at risk (predicted_failure=1). Unlisted drives are implicitly 0.
- Every predicted drive MUST include at least one evidence ref_id in "support".
- Every atomic claim MUST cite at least one ref_id.
- Keep it concise: focus on the top-risk subset, not all drives.
"""


def fleet_descriptive_user_prompt(payload: Dict[str, Any]) -> str:
    return f"""Task: Fleet-level descriptive analysis (cohort of drives).
Input JSON (payload): {payload}

Return EXACTLY ONE JSON object:
{{
  "task": "descriptive",
  "cohort_id": <string>,
  "summary": <text>,
  "fleet_patterns": [<text>, ...],
  "top_risk_drives": [<drive_id>, ...],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Summary should explain fleet health patterns and key SMART signals.
- "top_risk_drives" should be supported by cited evidence.
- Atomic claims must cite refs.
"""


def fleet_prescriptive_user_prompt(payload: Dict[str, Any]) -> str:
    return f"""Task: Fleet-level prescriptive analysis (cohort of drives).
Input JSON (payload): {payload}

Return EXACTLY ONE JSON object:
{{
  "task": "prescriptive",
  "cohort_id": <string>,
  "recommendations": [
    {{
      "action": <text>,
      "priority": <low|med|high>,
      "justification": <text>,
      "support": [<ref_id>, ...]
    }}
  ],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Provide fleet-appropriate operational actions (monitoring, migration, throttling, env controls).
- Cite IR refs for cohort-specific signals; cite literature refs for mechanisms/constraints.
- Atomic claims must cite refs.
"""


def fleet_whatif_user_prompt(payload: Dict[str, Any], scenario: str) -> str:
    return f"""Task: Fleet-level what-if analysis (cohort of drives).
Input JSON (payload): {payload}

Counterfactual scenario: {scenario}

Return EXACTLY ONE JSON object:
{{
  "task": "whatif",
  "cohort_id": <string>,
  "scenario": <text>,
  "analysis": <text>,
  "counterfactual_statements": [
    {{
      "statement": <text>,
      "variable": <text>,
      "delta": <number|null>,
      "effect": <text>,
      "effect_direction": <increase|decrease|unclear>,
      "evidence": [<ref_id>, ...]
    }}
  ]
}}

Rules:
- Every counterfactual statement MUST cite evidence refs.
- Use direction consistent with evidence when possible.
"""
