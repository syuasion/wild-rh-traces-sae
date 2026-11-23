# Wild Agent Reward-Hacking Traces + SAE Features (`v4_wild_full`)

Flagged multi-step agent steps, sparse autoencoder (SAE) features (`qwen3.5-2b-sae-bf16`), dense embeddings (`BAAI/bge-base-en-v1.5`), and Union Consensus taxonomy classifications from a category-free 2-stage scan over **738,619 purely wild multi-step agent steps** (`44,359` traces across 10 wild benchmarks; **zero** datasets constructed to elicit reward hacking).

## Summary

- **Scored corpus:** `738,619` wild steps (`44,359` traces) across 10 benchmarks (`DTap-Bench` benign split, `Terminal-Bench 2.0`, `Goodfire` `DeepSWE` / `SWE-bench Verified` / `τ³-bench` / `ImpossibleBench` / `ShoppingBench` [excluded after audit], `KernelBench` `Hard` / `CUDA` / `Mega`).
- **Flagged pool (`Option 1`):** **`1,186` steps** (`991` unique traces across 9 clean benchmarks), selected via 2-stage category-free Quail `AI.SCORE` (`Qwen3.8-27B-FP8`, `m24 = min(p2_rh_focused, p4_action_verifier)`: `m24 ≥ 0.15` wild, `≥ 0.35` ImpossibleBench, max 3 steps/trace, bare empty-reasoning `terminate`/`mark_task_complete`/`List MCP Tools` calls stripped).
- **Confirmed wild hacks (`Union Consensus`):** **`658 / 1,186` steps (`55.5%` precision)** across **`576` unique traces** spanning **9 organically discovered mechanisms** (`Fabricate missing data or outputs` 257, `Falsify status or unexecuted actions` 209, `Hardcode or overfit test cases` 52, `Bypass policy or tool restrictions` 35, `Substitute dummy or stub implementation` 32, `Tamper with tests or evaluation` 30, `Access hidden solutions or internals` 17, `Hijack runtime or harness hooks` 15, `Delegate to forbidden reference implementation` 11).

## Files

| File | Size | Description |
| --- | --- | --- |
| `data/flagged_steps.parquet` | 1.3 MB | All `1,186` flagged wild steps with full `task_description`, `reasoning`, `commands`, and Stage-1/2 scores (`p2`, `p4`, `p1`, `m24`). |
| `data/sae_features.parquet` | 594 KB | Sparse `qwen3.5-2b-sae-bf16` features (`32,768` dims, last-token pooling via Quail `AI.SAE_FEATURES`) for all `1,186` steps. |
| `data/embeddings.parquet` | 5.4 MB | L2-normalized 768-d `BAAI/bge-base-en-v1.5` CLS embeddings for all `1,186` steps. |
| `data/consensus_codes.parquet` | 292 KB | Quail `Qwen3.8-27B-FP8` `AI.CLASSIFY` intent & target labels + full probability distributions under the Union Consensus codebook. |
| `metadata/consensus_codebook.json` | 13 KB | Union Consensus codebook (13 intents [9 hack + 3 non-hack + `None of these`], 11 targets, definitions, include/exclude rules). |
| `metadata/five_metrics_summary.json` | 2.6 KB | 5 core clustering & taxonomy metrics across `BGE + K-Means`, `SAE + NMF`, `TF-IDF + NMF`, and `Random` (`n = 2` runs each). |
| `metadata/corpus_summary.json` | 3.4 KB | Full 10-benchmark corpus & flagged counts. |
| `metadata/filter_summary.json` | 0.9 KB | Exact Option 1 selection rules, thresholds, and artifact stripping counts. |

## Quick Start (Python)

```python
import pyarrow.parquet as pq
from scipy import sparse
import numpy as np

steps = pq.read_table("data/flagged_steps.parquet").to_pandas()
codes = pq.read_table("data/consensus_codes.parquet").to_pandas().rename(columns={"s.id": "step_id"})
df = steps.merge(codes[["step_id", "intent", "target"]], on="step_id", how="left")
print(df["intent"].value_counts())

# Load SAE features as a (1186, 32768) CSR sparse matrix aligned to steps["step_id"]
sae_tbl = pq.read_table("data/sae_features.parquet")
pos = {sid: i for i, sid in enumerate(sae_tbl["s.id"].to_pylist())}
sae_tbl = sae_tbl.take([pos[sid] for sid in steps["step_id"]])
col = sae_tbl["last_features"].combine_chunks()
flat = col.flatten()
sae_csr = sparse.csr_matrix(
    (np.maximum(flat.field("strength").to_numpy(), 0.0), flat.field("feature").to_numpy(), col.offsets.to_numpy()),
    shape=(len(steps), 32768),
)
print("SAE matrix:", sae_csr.shape, "nnz:", sae_csr.nnz)
```

## Schemas

- **`data/flagged_steps.parquet`** (`1,186` rows): `step_id`, `run_id`, `step_index`, `dataset`, `agent`, `task`, `task_description`, `reasoning`, `commands`, `p2`, `p4`, `p1`, `m24`, `hack_score`
- **`data/sae_features.parquet`** (`1,186` rows): `s.id` (`string`), `last_features` (`list<struct<feature: int32, strength: float>>`)
- **`data/embeddings.parquet`** (`1,186` rows): `step_id` (`string`), `embedding` (`list<double>`, len 768)
- **`data/consensus_codes.parquet`** (`1,186` rows): `s.id` (`string`), `intent`, `intent_probabilities`, `target`, `target_probabilities`

## Source Datasets

All traces originate from public HuggingFace benchmarks:
- [harborframework/terminal-bench-2-leaderboard](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard)
- [AI-Secure/DTap-Bench-Agent-Trajectories](https://huggingface.co/datasets/AI-Secure/DTap-Bench-Agent-Trajectories) (benign split only)
- [Goodfire/reward-hack-data](https://huggingface.co/datasets/Goodfire/reward-hack-data)
- [Infatoshi/kernelbench-hard-traces](https://huggingface.co/datasets/Infatoshi/kernelbench-hard-traces), [kernelbench-cuda-traces](https://huggingface.co/datasets/Infatoshi/kernelbench-cuda-traces), [kernelbench-mega-traces](https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces)
