# Knowledge Graph Schema — Raman-informed 2D Material Exploration Agent

This document defines the knowledge graph (KG) that the exploration agent navigates. The agent is **read-only over structures**: it does not generate or modify materials. It only retrieves, ranks, and explains.

The KG is a **typed multigraph**:

- **Nodes** = materials (one node per entry in the DB).
- **Edges** = typed relations between two materials, each carrying a `relation_type` and a scalar `weight` in `[0, 1]`.
- Multiple edges of different types between the same pair are allowed and **must** be preserved — they are the provenance the agent uses to explain "why these two materials are related".

---

## 1. Node attributes

Every node stores at minimum:

| field                 | type              | notes                                                     |
|-----------------------|-------------------|-----------------------------------------------------------|
| `material_id`         | str               | unique key                                                |
| `formula`             | str               | e.g. `MoSSe`                                              |
| `prototype`           | str               | structural prototype label (e.g. `2H`, `1T`, `1T'`, ...)  |
| `family`              | str               | coarse chemical family (e.g. `TMDC`, `MXene`, ...)        |
| `metal`               | str               | metal element symbol                                      |
| `chalcogen_top`       | str               | top-layer chalcogen                                       |
| `chalcogen_bottom`    | str               | bottom-layer chalcogen                                    |
| `is_janus`            | bool              | `chalcogen_top != chalcogen_bottom`                       |
| `raman_spectrum`      | np.ndarray[4000]  | raw / preprocessed Raman intensities                      |
| `structure_embedding` | np.ndarray[d_s]   | pretrained structure graph embedding (given)              |
| `raman_embedding`     | np.ndarray[d_r]   | pretrained Raman embedding (given)                        |

> Derived fields like `is_janus` must be computed once at ingest time so
> rules below can assume they exist.

---

## 2. Design principles

Before listing the rules, the conventions they all follow:

1. **No self-loops.** Never add an edge from a node to itself.
2. **Undirected.** Every rule below is symmetric. Store each pair once.
3. **Specificity beats breadth.** If a more specific rule already connects two nodes with a higher weight, the less specific rule's edge is still stored (for explanation), but aggregation uses `max` per pair, not `sum`.
4. **Embeddings are the single source of truth for "similarity".** Raw 4000-dim Raman cosine is **not** used as a separate edge — the Raman embedding already encodes it and using both double-counts the same signal. Raw spectra are kept on the node for visualization and for the agent's downstream reasoning, not as edges.
5. **Similarity edges are sparsified by percentile, not by absolute threshold.** Absolute cosine thresholds don't transfer across datasets or embedding versions. We keep the top-`p%` of pairs per rule.
6. **Typed edges are preserved** even when redundant. The agent uses the *set* of edge types between a pair as the explanation.

---

## 3. Edge rules

There are **two groups**: categorical (rules R1–R4) and embedding-based (rules R5–R6). Janus relationships are encoded as an **edge attribute**,not a separate edge type, to avoid overlap with R2.

### Group A — Categorical / compositional edges

These come from discrete node attributes. They are cheap, exact, and highly interpretable.

#### R1. `structure_neighbor` — weight `0.90`

The strongest categorical edge: same prototype, one atomic substitution away.

**Conditions (all must hold):**
- `prototype[a] == prototype[b]`
- `composition_distance(a, b) == 1`

where
```
composition_distance(a, b) =
    int(metal[a]            != metal[b])
  + int(chalcogen_top[a]    != chalcogen_top[b])
  + int(chalcogen_bottom[a] != chalcogen_bottom[b])
```

**Edge attributes:**
- `relation_type = "structure_neighbor"`
- `weight = 0.90`
- `is_janus_pair = (is_janus[a] != is_janus[b])`  *(replaces old R6)*

Rationale: R6 "janus_analog" in the original list was a strict subset of this rule (same prototype, shared chalcogen, different `is_janus` → composition_distance is 1 or 2). We fold it in as an attribute so the agent can still filter by "Janus ↔ non-Janus analog" without creating a second edge that the aggregator would have to dedupe.

> **Check before implementing:** verify whether your DB assigns the *same*
> prototype label to a Janus material and its non-Janus parent (e.g.
> MoS2 `2H` vs MoSSe `2H-Janus`). If they get different prototype strings,
> this rule will miss Janus analogs — in that case either normalize the
> prototype field or add a separate rule `R1b` that relaxes `prototype`
> equality to "prototype base equality" (stripping the `-Janus` suffix).

#### R2. `same_prototype` — weight `0.80`

**Conditions:**
- `prototype[a] == prototype[b]`
- R1 does **not** already fire for `(a, b)` *(i.e. composition_distance >= 2)*

**Weight:** `0.80`.

Keeping R1's pairs out of R2 prevents two edges with the same categorical information from coexisting on the same pair.

#### R3. `shared_chalcogen` — weight `0.60` or `0.70`

Chalcogen composition overlap, independent of prototype.

```
left  = {chalcogen_top[a], chalcogen_bottom[a]}
right = {chalcogen_top[b], chalcogen_bottom[b]}
k = len(left & right)
```

- If `k == 0`: no edge.
- If `k >= 1`: `weight = 0.50 + 0.10 * k`, so `k=1 -> 0.60`, `k=2 -> 0.70`.

> Note: `k == 2` is only reachable when **both** materials are Janus with
> the *same* unordered pair of chalcogens. Non-Janus materials have set
> size 1, so intersecting two non-Janus materials gives at most `k == 1`.
> This is the intended semantics — no change needed, but document it.

#### R4. `same_family` — weight `0.50` *(downgraded)*

**Conditions:**
- `family[a] == family[b]`
- R1, R2 do not fire for `(a, b)`

Because `same_prototype` usually implies `same_family`, we only emit R4 when nothing stronger already connects the pair. Weight reduced from the originally proposed `0.65` to `0.50` to reflect that `family` is the *coarsest* categorical signal.


---

### Group B — Embedding-based similarity edges

These use the two pretrained embeddings you already have. They capture **continuous** similarity that the categorical rules can't express.

#### General recipe (applies to R5 and R6)

For embedding vectors `e_a`, `e_b`:

1. L2-normalize once at ingest: `e_hat = e / ||e||_2`.
2. Compute cosine: `sim = e_hat_a . e_hat_b` (range `[-1, 1]`; in practice `[0, 1]` for these embeddings).
3. **Sparsify by percentile, not by absolute threshold.** For each rule, keep only the **top `p` percent** of all unordered pairs, globally.
   Recommended defaults:
   - `p_structure_embedding = 1.0` (top 1% of pairs)
   - `p_raman_embedding     = 1.0` (top 1% of pairs)
4. The **edge weight is the cosine similarity itself**, clipped to `[0, 1]`. No rescaling — the agent can compare weights across R5/R6 directly because both are cosine in `[0, 1]`.

Expose the percentiles in `cfg` so they are tunable:
```yaml
kg:
  structure_embedding_top_percent: 1.0
  raman_embedding_top_percent: 1.0
```

Optionally also support absolute floors as a safety net (e.g. never keep an edge with `sim < 0.5` even if it falls in the top percentile on a small dataset):
```yaml
kg:
  structure_embedding_min_sim: 0.5
  raman_embedding_min_sim: 0.5
```

#### R5. `structure_embedding_neighbor`

- **Embedding:** `structure_embedding` (pretrained structure graph embedding, already available per node).
- **Score:** cosine similarity on L2-normalized vectors.
- **Sparsification:** keep top `p_structure_embedding` percent of all pairs, subject to optional absolute floor.
- **Edge:**
  - `relation_type = "structure_embedding_neighbor"`
  - `weight       = sim_structure   in [0, 1]`

Captures structural similarity that categorical rules miss
(e.g. different prototype but geometrically / topologically similar).

#### R6. `raman_embedding_neighbor`

- **Embedding:** `raman_embedding` (pretrained Raman embedding, already available per node).
- **Score:** cosine similarity on L2-normalized vectors.
- **Sparsification:** keep top `p_raman_embedding` percent of all pairs, subject to optional absolute floor.
- **Edge:**
  - `relation_type = "raman_embedding_neighbor"`
  - `weight       = sim_raman   in [0, 1]`

Captures spectroscopic similarity. Because the Raman embedding was trained on the full spectrum, this edge **replaces** the original `raman_neighbor` rule that used raw 4000-dim cosine — using both double-counts the same signal.

---

## 4. Aggregation for the agent

When the agent needs a single similarity score between two nodes `(a, b)`
(e.g. for ranking retrieval results), aggregate across the edges present between them:

```
score(a, b) = max over edges e in E(a,b) of weight(e)
```

Always return the **list of contributing edge types** alongside the score so the agent can produce an explanation like:

> MoSe2 is related to MoS2 via `structure_neighbor` (0.90) and
> `raman_embedding_neighbor` (0.87). The pair is not a Janus pair.

If a blended score is needed later (for learning-to-rank), prefer learning per-edge-type weights on a held-out task rather than hand-tuning a weighted sum now.

---

## 5. Summary table

| Rule | Relation                        | Trigger                                                 | Weight                |
|------|---------------------------------|---------------------------------------------------------|-----------------------|
| R1   | `structure_neighbor`            | same prototype & composition_distance == 1              | `0.90`                |
| R2   | `same_prototype`                | same prototype & composition_distance >= 2              | `0.80`                |
| R3   | `shared_chalcogen`              | `|chalcogens(a) & chalcogens(b)| >= 1`                  | `0.50 + 0.10 * k`     |
| R4   | `same_family`                   | same family & not R1/R2                                 | `0.50`                |
| R5   | `structure_embedding_neighbor`  | cosine(structure_embedding) in top `p_s%`               | cosine sim, clipped   |
| R6   | `raman_embedding_neighbor`      | cosine(raman_embedding) in top `p_r%`                   | cosine sim, clipped   |

**Dropped vs. original proposal:**
- `same_metal` — too dense, information already in `structure_neighbor` / node attr.
- `raman_neighbor` (raw 4000-dim cosine) — redundant with R6 (`raman_embedding_neighbor`).
- `janus_analog` (standalone edge) — folded into R1 as `is_janus_pair` attribute.