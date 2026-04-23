KORAL Stage I - Literature KG Builder (Paper → KG → TTL)
=======================================================

Stage I builds a Literature Knowledge Graph (LitKG) from SSD research papers.
It is designed to produce evidence-backed, taxonomy-aligned assertions that Stage II can retrieve later.

What Stage I does
-----------------
1) Iterates over papers in a folder (supported: .pdf, .txt, .md)
2) Extracts clean text (PDF via PyPDF2 when available)
3) Calls an LLM (default: GPT-4o) using a strict JSON prompt
4) Validates/matches entities against the SSD taxonomy (taxonomy.json)
5) Produces per-paper artifacts:
     - <paper_slug>.kg.json  (strict JSON extraction output)
     - <paper_slug>.ttl      (RDF/Turtle graph with provenance)
6) Merges per-paper TTLs into a Global Knowledge Graph:
     - global_knowledge_graph.ttl
   The merge is append-only (no overwrite); provenance is preserved.

Where to put it in your repo
----------------------------
Your repo layout:
  KORAL/
    stage_I/
      out/
      __init__.py
      ssd_cot_prompt.txt
      ssd_kg_pipeline.py
      taxonomy.json
    stage_II/
    data_preparation/
    dataset/

Stage I lives in:
  <repo_root>/stage_I/

How to run
----------
1) Install dependencies (Stage I):
   pip install openai rdflib PyPDF2 python-dotenv

2) Export OpenAI key:
   export OPENAI_API_KEY="sk-..."

3) IMPORTANT: configure prompt paths
   The Stage I script defaults to:
     prompts/ssd_cot_prompt.txt
     prompts/ssd_prompt_addenda_auto.txt

   But in this repo, the prompt is stored in stage_I/ssd_cot_prompt.txt.
   So set:

   export KG_PROMPT_PATH="stage_I/ssd_cot_prompt.txt"
   export KG_PROMPT_ADDENDA_PATH="stage_I/out/ssd_prompt_addenda_auto.txt"

   Alternative:
   - create <repo_root>/prompts/ and copy the prompt there, keeping defaults.

4) Prepare a papers folder
   Create a folder containing SSD papers as PDF/TXT/MD:

   dataset/papers/
     paper1.pdf
     paper2.pdf
     report.txt
     ...

5) Run Stage I
   From repo root:

   python stage_I/ssd_kg_pipeline.py \
     --papers_dir dataset/papers \
     --taxonomy stage_I/taxonomy.json \
     --out_dir stage_I/out \
     --model gpt-4o

Notes on input folder
---------------------
- Stage I scans only the top-level files in --papers_dir:
    *.pdf, *.txt, *.md
- If you keep papers nested in subfolders, move/copy them into one flat folder
  or modify the script to recurse.

Outputs
-------
Stage I writes outputs into --out_dir:

stage_I/out/
  <paper_slug>.kg.json            # strict JSON extraction per paper
  <paper_slug>.ttl                # per-paper LitKG (RDF/Turtle)
  global_knowledge_graph.ttl      # merged KG across all papers and runs

Additionally:
- taxonomy.json can be updated in-place if new concepts are proposed and accepted
  by the pipeline’s controlled insertion step (keeps vocabulary evolving).
- A prompt addenda file can be written (path set by KG_PROMPT_ADDENDA_PATH) to
  help future extractions stay consistent with previously seen terminology.

Environment variables (advanced)
--------------------------------
The Stage I pipeline supports configuration via env vars:

- KG_LLM_MODEL               default model name (default: gpt-4o)
- KG_BASE_URI                base namespace for instances
- KG_TAXONOMY_URI             base namespace for taxonomy classes
- KG_ASSERTION_URI            base namespace for assertion nodes
- KG_PROMPT_PATH              path to the extraction prompt text file
- KG_PROMPT_ADDENDA_PATH      path for auto-generated prompt addenda

Troubleshooting
---------------
1) "No papers found"
   - Verify --papers_dir contains .pdf/.txt/.md files (not nested in subfolders)

2) PDF extraction errors
   - Ensure PyPDF2 is installed:
       pip install PyPDF2

3) Prompt not found
   - Ensure KG_PROMPT_PATH points to:
       stage_I/ssd_cot_prompt.txt

4) Merge behavior
   - global_knowledge_graph.ttl is merged with any previous existing file in --out_dir.
     If you want a clean rebuild, delete stage_I/out/global_knowledge_graph.ttl first.

What Stage II expects from Stage I
----------------------------------
Stage II uses:
- global_knowledge_graph.ttl  (LitKG evidence base)
- taxonomy.json               (shared vocabulary / anchors)

Typical practice after Stage I:
  cp stage_I/out/global_knowledge_graph.ttl global_knowledge_graph.ttl
  cp stage_I/taxonomy.json taxonomy.json

(or update Stage II config to point to stage_I/out/global_knowledge_graph.ttl)
