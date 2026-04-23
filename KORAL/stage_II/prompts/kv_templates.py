#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates for KV-SSD diagnosis (Stage II KV path).

Four tasks:
  - diagnose  : descriptive explanation of current KV-SSD behavior
  - attribute : causal attribution of observed anomaly to a KV-FTL mechanism
  - prescribe : actionable tuning knobs (inline threshold, CMT size, GC policy)
  - whatif    : counterfactual reasoning over KV-SSD internals
"""

from __future__ import annotations
import json
from typing import Any, Dict


def kv_system_prompt(base_cot: str | None = None) -> str:
    core = """You are KORAL Stage II for KV-SSD internal performance diagnosis.
You reason about KV-FTL behaviors — mapping tables, inline vs. regular entries,
CMT (Cached Mapping Table) dynamics, translation-page pressure, inline-to-regular
transitions, mapping-entry churn — and how they shape observable performance
(tail latency, throughput, read/write amplification).

You MUST return a single valid JSON object and nothing else. No markdown fences.
If something cannot be grounded in the provided IR or Literature refs, say so in
`uncertainty` and prefer `null` over guessing.

Evidence IDs accepted in `support`:
  - KV_SAMPLE, KV_DEVICE, KV_<signal>   (from the sample's Intermediate Representation)
  - LIT_<n>                             (from the retrieved Literature KG)

Do NOT invent other IDs.
"""
    if base_cot and base_cot.strip():
        return core + "\n\nAdditional guidance:\n" + base_cot.strip()
    return core


def _compact_payload(sample: Dict[str, Any]) -> str:
    # Keep the payload small; dump as JSON once.
    return json.dumps(sample, ensure_ascii=False, default=str)


def diagnose_user_prompt(sample: Dict[str, Any]) -> str:
    return f"""Task: Diagnose the current KV-SSD behavior.

Sample (IR + observations + literature): {_compact_payload(sample)}

Produce JSON:
{{
  "task": "diagnose",
  "sample_id": <string>,
  "summary": <2-4 sentence narrative citing KV-FTL mechanisms>,
  "observed_signals": [<short phrase for each signal used>],
  "likely_mechanisms": [
    {{"mechanism": <"CMT Hit Rate Dynamics"|"Inline-to-Regular Transition"|"Mapping Entry Churn"|"Translation Page Pressure"|"KV Separation"|"Compaction"|"GC">,
      "why": <short text>,
      "support": [<ref_id>, ...]}}
  ],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}],
  "uncertainty": <text|null>
}}

Rules:
- Prefer citing KV_<signal> refs when the claim is about this sample's behavior.
- Cite LIT_<n> refs when invoking general KV-SSD mechanisms.
- Every atomic claim MUST have at least one support ID.
"""


def attribute_user_prompt(sample: Dict[str, Any]) -> str:
    return f"""Task: Causally attribute the observed performance anomaly to KV-FTL mechanisms.

Sample: {_compact_payload(sample)}

Produce JSON:
{{
  "task": "attribute",
  "sample_id": <string>,
  "primary_cause": {{
    "mechanism": <string>,
    "confidence": <0..1>,
    "explanation": <short text>,
    "support": [<ref_id>, ...]
  }},
  "contributing_causes": [
    {{"mechanism": <string>, "explanation": <short text>, "support": [<ref_id>, ...]}}
  ],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Rank causes by how directly the IR signals implicate them.
- If the IR is insufficient to distinguish causes, set primary_cause.confidence <= 0.5.
"""


def prescribe_user_prompt(sample: Dict[str, Any]) -> str:
    return f"""Task: Recommend KV-SSD tuning actions.

Sample: {_compact_payload(sample)}

Produce JSON:
{{
  "task": "prescribe",
  "sample_id": <string>,
  "recommendations": [
    {{"action": <text>,
      "priority": <"low"|"med"|"high">,
      "target_signal": <signal name, e.g. "cmt_hit_rate">,
      "expected_effect": <short text>,
      "justification": <text>,
      "support": [<ref_id>, ...]}}
  ],
  "atomic_claims": [{{"claim": <text>, "support": [<ref_id>, ...]}}]
}}

Rules:
- Prefer KV-FTL-level actions (increase inline threshold, resize CMT, tune GC trigger,
  switch KV-separation policy) over generic advice.
- Each recommendation needs at least one support ID.
"""


def whatif_user_prompt(sample: Dict[str, Any], scenario: str) -> str:
    return f"""Task: What-if analysis for KV-SSD internals.

Sample: {_compact_payload(sample)}
Counterfactual scenario: {scenario}

Produce JSON:
{{
  "task": "whatif",
  "sample_id": <string>,
  "scenario": <text>,
  "analysis": <short text>,
  "counterfactual_statements": [
    {{"statement": <text>,
      "variable": <signal name>,
      "delta": <number|null>,
      "effect": <text>,
      "effect_direction": <"increase"|"decrease"|"unclear">,
      "evidence": [<ref_id>, ...]}}
  ]
}}

Rules:
- Each counterfactual statement MUST include at least one evidence ID.
- effect_direction must be consistent with the cited evidence when possible.
"""
