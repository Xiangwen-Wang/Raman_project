# Action Reward — Raman-informed 2D Material Exploration Agent

This document defines how the **value / reward of an exploration action** is computed for the Raman-informed 2D material exploration agent.

It answers:

* Why is an action considered good?
* How good is it numerically?
* When should the agent retrieve, expand, rank, query DFT, update knowledge, or stop?

The reward is designed for an agent that is **read-only over structures**.
The agent does not modify or generate materials. It explores an existing or expandable structure–Raman knowledge graph by retrieving, expanding, ranking, querying, updating, and stopping.

The reward must be:

* **Quality-first**: finding better Raman matches matters more than finding many candidates.
* **Knowledge-aware**: useful new information is rewarded.
* **Cost-sensitive**: expensive DFT queries are penalized.
* **Redundancy-resistant**: repeated or uninformative actions are discouraged.
* **Bounded on positive terms**: positive rewards are capped so no term dominates accidentally.
* **Action-specific**: retrieve / expand / rank / query / update / stop actions are evaluated differently.

> **How this doc relates to the KG schema.** See `docs/kg_schema.md` for the graph the agent navigates. Every `match_score` referenced here is computed via the similarity primitives in Section 1 below, applied to the Raman target the user supplied and the candidate material stored on a KG node.

---

## 1. Base similarity primitives

### 1.1 `cosine_similarity(xs, ys) -> float`

Standard cosine similarity:

```python
dot(xs, ys) / (norm(xs) * norm(ys))
```

Returns `0.0` if:

* either input is empty
* lengths differ
* either norm is `0`

`embedding_similarity(e_a, e_b)` is an alias for `cosine_similarity()` on pre-normalized embeddings.

---

### 1.2 `peak_similarity(a, b, sigma, tolerance=None) -> float`

`peak_similarity()` measures Raman peak-level similarity.

Inputs `a` and `b` each contain a list of Raman peaks:

```python
(wavenumber, intensity)
```

#### Step 1 — Intensity normalization

For each spectrum:

```python
w_a[i] = intensity_a[i] / sum(intensity_a)
w_b[j] = intensity_b[j] / sum(intensity_b)
```

Fallback to uniform weights when:

* number of intensities does not match number of peaks
* intensity sum is `<= 0`
* any intensity is negative or NaN

After normalization:

```python
sum(w_a) == 1
sum(w_b) == 1
```

#### Step 2 — Decay kernel

For a wavenumber difference `delta`:

```python
decay(delta) = exp(-0.5 * (delta / sigma) ** 2)   if |delta| <= tolerance
             = 0.0                                 otherwise
```

The default tolerance is coupled to `sigma`:

```python
tolerance = 3 * sigma
```

Users may override this value. A warning should be issued if:

```python
tolerance > 5 * sigma        # wasted computation (decay already ~0)
tolerance < sigma            # creates a sharp matching cliff
```

#### Step 3 — Bidirectional nearest-neighbor matching

For each peak `i` in spectrum `a`, find the nearest peak `j*` in spectrum `b`:

```python
contrib_a_to_b[i] = decay(delta_ij*) * sqrt(w_a[i] * w_b[j*])
```

The **geometric mean** `sqrt(w_a * w_b)` is used instead of `min(w_a, w_b)`.

Rationale: `min(w_a, w_b)` acts like a strict AND gate and can overly suppress a good peak-position match when one side has weaker intensity.
The geometric mean degrades more smoothly.

Compute the reverse direction symmetrically:

```python
contrib_b_to_a[j] = decay(delta_ji*) * sqrt(w_b[j] * w_a[i*])
```

Then:

```python
sim_ab = sum_i contrib_a_to_b[i]
sim_ba = sum_j contrib_b_to_a[j]

peak_sim_raw = 0.5 * (sim_ab + sim_ba)
```

#### Step 4 — Clamp

```python
peak_similarity = clip(peak_sim_raw, 0.0, 1.0)
```

The clamp is **required, not defensive**. Bidirectional nearest-neighbor matching is not one-to-one: multiple peaks in one spectrum may match the same peak in the other. In rare many-to-one cases the raw score can exceed `1.0`.

This is acceptable for v1.

Upgrade path: replace nearest-neighbor matching with Hungarian one-to-one assignment if many-to-one false matches become a problem.

---

### 1.3 `raman_similarity(current, target, mode, alpha=0.3) -> float`

`raman_similarity()` selects between full-spectrum similarity, peak-level similarity, and hybrid similarity. **This is the function that produces every `match_score` referenced later in Sections 5, 8, 9, 10.**

First check data availability:

```python
use_full  = bool(current.full_spectrum and target.full_spectrum)
use_peaks = bool(current.peaks and target.peaks)
```

| mode     | output                                                                |
| -------- | --------------------------------------------------------------------- |
| `full`   | `cosine_similarity(full_a, full_b)` if full spectra exist, else `0.0` |
| `peaks`  | `peak_similarity(peaks_a, peaks_b, sigma)` if peaks exist, else `0.0` |
| `hybrid` | weighted combination of full and peak similarity                      |

Hybrid mode:

```python
if use_full and use_peaks:
    return alpha * full_sim + (1 - alpha) * peak_sim
elif use_peaks:
    return peak_sim
elif use_full:
    return full_sim
else:
    return 0.0
```

Default `alpha = 0.3`.

Rationale: Raman peaks carry more direct chemical and structural information. The full spectrum adds broad-shape information that overlaps with the peak representation. Therefore peak similarity is weighted more heavily than full-spectrum cosine.

---

## 2. Information gain

```python
information_gain(new_candidates, high_value_hits, cfg) -> float
```

Information gain rewards actions that discover new useful candidates.

The reward is `0.0` unless `new_candidates > 0`.

### 2.1 Saturation caps

```python
scaled_new  = min(new_candidates,   cfg.info_gain_new_cap)  / cfg.info_gain_new_cap
scaled_hits = min(high_value_hits,  cfg.info_gain_hits_cap) / cfg.info_gain_hits_cap
```

Default caps: `info_gain_new_cap = 10`, `info_gain_hits_cap = 5`.

### 2.2 Quality before quantity

```python
gain = cfg.info_gain_weight * (0.3 * scaled_new + 0.7 * scaled_hits)
```

Default `info_gain_weight = 0.5`.

`high_value_hits` is defined as:

```python
high_value_hits =
    |{ c in new_candidates : c.match_score >= cfg.high_value_similarity }|
```

Default `high_value_similarity = 0.80`.

`high_value_hits` is a subset of `new_candidates`. A new high-quality candidate is therefore rewarded twice — once for being new, once for being high-value. This is intentional: the agent should prefer useful new materials over merely numerous new materials.

---

## 3. Diversity and novelty bonus

Information gain alone can still reward the agent for repeatedly exploring the same local cluster (e.g. five new materials all within `0.01` cosine of each other in structure-embedding space).

To avoid this, add a diversity and novelty bonus.

```python
diversity_novelty_bonus(new_candidates, seen_candidates, cfg)
```

This term has two parts:

1. **Internal diversity** among the new candidates.
2. **Novelty** relative to already seen or already visited candidates.

---

### 3.1 Distance metric

Use structure embedding cosine distance:

```python
distance(a, b) = clip(
    1 - cosine_similarity(a.structure_embedding, b.structure_embedding),
    0.0, 1.0,
)
```

Why clip: raw `1 - cosine` has range `[0, 2]` when cosine ∈ `[-1, 1]`.
For the L2-normalized embeddings produced by the pretrained structure encoder, cosine is almost always in `[0, 1]`, but we clip for numerical safety.

---

### 3.2 Internal diversity

```python
if len(new_candidates) >= 2:
    internal_diversity = mean_pairwise_distance(new_candidates)
else:
    internal_diversity = 0.0
```

---

### 3.3 Novelty relative to visited space

For each new candidate, compute its distance to the closest already-seen candidate:

```python
novelty(c) = min distance(c, s) for s in seen_candidates
```

Then:

```python
relative_novelty = mean_c novelty(c)
```

If `seen_candidates` is empty:

```python
relative_novelty = 0.0
```

because early-stage discovery is already rewarded by information gain.

---

### 3.4 Final diversity-novelty bonus

```python
diversity_novelty_bonus = cfg.diversity_weight * (
    0.5 * internal_diversity
  + 0.5 * relative_novelty
)
```

Default `diversity_weight = 0.2`. Set to `0.0` to disable.

This term encourages the agent to explore structurally different regions instead of repeatedly expanding the same local cluster.

---

## 4. General exploration reward

For most non-terminal exploration actions, the reward is:

```python
reward = cfg.w_focus * delta_focus
       + cfg.w_pool  * delta_pool
       + information_gain
       + diversity_novelty_bonus
       - action_cost
       - redundancy_penalty
```

where `action_cost` is looked up in the per-action table of Section 7, not a single global constant.

This base formula applies to:

* `retrieve`
* `expand`
* `query_dft_raman` (with additional terms, see Section 10)
* `update_knowledge` (with a different formulation, see Section 11)

`rank` and `stop` use their own formulas (Sections 8 and 9).

---

## 5. Improvement terms

### 5.1 Focus-set improvement

The focus set is the current set of candidates most relevant to the user's Raman objective.

```python
before_focus_best = max match_score in focus_set before action
after_focus_best  = max match_score in focus_set after action
```

where `match_score = raman_similarity(candidate, target, mode, alpha)` from Section 1.3.

Then:

```python
delta_focus = after_focus_best - before_focus_best
```

**Clamping depends on action type**:

```python
if action_type in ["prune", "narrow", "filter"]:
    # do NOT clamp — pruning a good candidate should hurt
    delta_focus = after_focus_best - before_focus_best
else:
    # growing actions only gain
    delta_focus = max(after_focus_best - before_focus_best, 0.0)
```

Default `w_focus = 1.0`.

---

### 5.2 Pool improvement

The pool is the broader global candidate set.

```python
before_pool_best = max match_score in candidate_pool before action
after_pool_best  = max match_score in candidate_pool after action
delta_pool       = after_pool_best - before_pool_best
```

Clamp using the same action-type rule as 5.1:

```python
if action_type in ["prune", "narrow", "filter"]:
    delta_pool = after_pool_best - before_pool_best
else:
    delta_pool = max(after_pool_best - before_pool_best, 0.0)
```

Default `w_pool = 0.5`. The agent is primarily rewarded for improving the focus set; global pool improvement is a secondary bonus.

---

### 5.3 Optional top-k smoothing (focus only)

When the best candidate is already close to `1.0`, `delta_focus` saturates to `~0` and the agent loses learning signal, even if it has meaningfully raised the 2nd–5th ranked candidates.

To avoid this, optionally use top-k smoothing on the **focus set only**:

```python
delta_focus_topk =
    mean(top_k_scores_after_focus) - mean(top_k_scores_before_focus)
```

Defaults `use_topk_delta = false`, `topk_delta_k = 5`.

If enabled:

```python
delta_focus = delta_focus_topk
```

Apply the same action-type clamping as 5.1.

**Pool-side smoothing is intentionally not offered**: the pool is coarser and large, so top-k on pool adds noise without new signal. 
Revisit if the pool is small (<200 candidates).

---

## 6. Redundancy penalty

Repeated actions should become increasingly expensive.

Define:

```python
repeat_count =
    number of previous times the same (action_type, target_hash)
    has been issued in the current session
```

First issuance: `repeat_count = 0`. Second: `1`. Etc.

To keep the penalty bounded:

```python
effective_repeat_count = min(repeat_count, cfg.repeat_count_cap)
redundancy_penalty     = cfg.redundancy_penalty * effective_repeat_count
```

Defaults: `redundancy_penalty = 0.1`, `repeat_count_cap = 5`. Maximum redundancy cost is therefore `0.5`.

A single accidental repeat is a mild warning; a sustained loop is bounded-expensive (the cap prevents any single action from producing unboundedly negative reward, which can destabilize learning).

**Stop is terminal and not subject to redundancy** — there is no "second stop" in a session.

---

## 7. Action cost

Each action type has its own cost. Use a lookup table rather than a single constant:

```yaml
action_costs:
  retrieve: 0.02
  expand: 0.03
  rank: 0.01
  query_dft_raman: 0.20
  update_knowledge: 0.01
  stop: 0.00
```

Rationale:

* database retrieval is cheap
* graph expansion is cheap to moderate
* ranking is cheap
* DFT query is expensive
* knowledge update is cheap
* stopping should not itself be penalized (early-stop is handled in 9)

When the reward formulas in Sections 4, 8, 10, 11 reference `action_cost`, substitute the value from this table.

If runtime or compute cost differs by more than ~3× within a single action type, pass an override per-call.

---

## 8. Rank action reward

The general exploration-reward formula underestimates `rank` actions, because ranking does not add new candidates or change the global pool.

A rank action should be rewarded if it improves the quality of the candidates surfaced to the agent or user.

For a ranked list, define:

```python
topk_quality = mean(match_score of top-k ranked candidates)
rank_delta   = topk_quality_after - topk_quality_before
```

**Do not clamp `rank_delta` to `0`.** A re-ranking that surfaces worse candidates should produce negative reward — that is what teaches the agent ranking is not free.

```python
rank_reward = cfg.w_rank * rank_delta
            - action_cost         # = action_costs["rank"] = 0.01
            - redundancy_penalty
```

Defaults `w_rank = 0.5`, `rank_topk = 5`.

Optional uncertainty-aware rank reward:

```python
rank_reward = cfg.w_rank * rank_delta
            - cfg.w_rank_uncertainty * mean_uncertainty_topk
            - action_cost
            - redundancy_penalty
```

Default `w_rank_uncertainty = 0.0`. Set above zero only if uncertainty estimates are reliable.

---

## 9. Stop action reward

Stop is **terminal** — it ends the episode. It needs its own reward so the agent learns *when* to stop, not just *what* to explore.

Stopping is good when:

* the top candidates already match the Raman target well
* uncertainty is low
* further exploration is unlikely to improve the result
* the agent avoids unnecessary DFT or repeated search

Define:

```python
topk_match       = mean(match_score of top-k final candidates)
topk_uncertainty = mean(uncertainty of top-k final candidates)
```

Base stop reward:

```python
stop_reward = cfg.w_stop_quality       * topk_match
            - cfg.w_stop_uncertainty   * topk_uncertainty
            - cfg.stop_cost
```

Defaults `w_stop_quality = 1.0`, `w_stop_uncertainty = 0.3`, `stop_topk = 5`, `stop_cost = 0.0`.

Threshold-based adjustment:

```python
if topk_match >= cfg.stop_match_threshold:
    stop_reward += cfg.stop_success_bonus
else:
    stop_reward -= cfg.stop_early_penalty
```

Defaults `stop_match_threshold = 0.80`, `stop_success_bonus = 0.2`, `stop_early_penalty = 0.2`.

No redundancy penalty applies (stop is terminal).

---

## 10. DFT query reward

`query_dft_raman` is an expensive but important action. It should be rewarded when it:

* validates a high-potential candidate
* reduces uncertainty
* fills a knowledge gap
* improves Raman matching
* creates useful new graph edges or updates existing ones

The query uses the general exploration reward plus a DFT-specific uncertainty-reduction term:

```python
dft_info_reward =
    cfg.w_uncertainty_reduction
  * max(uncertainty_before - uncertainty_after, 0.0)
```

Default `w_uncertainty_reduction = 0.3`. Clamped non-negative: a DFT query that **increases** reported uncertainty (e.g. DFT contradicts the prior) should not be rewarded, but also should not be punished by this term — it is punished via the missing `delta_focus / delta_pool` improvement and the high `action_cost`. (Treating "surprise" as informative is deferred to v2.)

Final DFT query reward:

```python
reward = cfg.w_focus * delta_focus
       + cfg.w_pool  * delta_pool
       + information_gain
       + diversity_novelty_bonus
       + dft_info_reward
       - action_costs["query_dft_raman"]     # = 0.20 by default
       - redundancy_penalty
```

> Note on double counting: `information_gain` rewards *discovering new
> candidates*; `dft_info_reward` rewards *becoming more confident in
> existing candidates*. These are different signals and both belong in
> the DFT formula.

---

## 11. Update knowledge reward

`update_knowledge` adds new Raman results or updated relations back into the knowledge graph.

This action is cheap but should be rewarded only if it actually improves the knowledge state. An "update" that adds nothing should produce negative reward (it still costs `action_cost + redundancy_penalty`).

Positive signals:

* new Raman data added
* new Raman embedding computed
* new structure–Raman edge created
* uncertainty reduced
* candidate ranking updated
* symbolic or embedding edge weights updated

Reward:

```python
update_reward = cfg.w_knowledge_update     * kg_quality_gain
              + cfg.w_uncertainty_reduction * uncertainty_reduction
              - action_costs["update_knowledge"]    # = 0.01
              - redundancy_penalty
```

Default `w_knowledge_update = 0.3`.

For v1, approximate:

```python
kg_quality_gain = min(
    n_new_or_updated_edges / cfg.kg_update_edge_cap,
    1.0,
)
```

Default `kg_update_edge_cap = 10`.

v2 extension: weight edges by their KG-schema weight (see
`docs/kg_schema.md`) so that adding one `structure_neighbor` (0.90)
counts for more than adding one `same_family` (0.50).

---

## 12. Final reward by action type

### 12.1 Retrieve / expand

```python
reward = cfg.w_focus * delta_focus
       + cfg.w_pool  * delta_pool
       + information_gain
       + diversity_novelty_bonus
       - action_cost
       - redundancy_penalty
```

### 12.2 Rank

```python
reward = cfg.w_rank * rank_delta
       - action_cost
       - redundancy_penalty
```

Optional:

```python
reward -= cfg.w_rank_uncertainty * mean_uncertainty_topk
```

### 12.3 Query DFT-Raman

```python
reward = cfg.w_focus * delta_focus
       + cfg.w_pool  * delta_pool
       + information_gain
       + diversity_novelty_bonus
       + cfg.w_uncertainty_reduction * uncertainty_reduction
       - action_costs["query_dft_raman"]
       - redundancy_penalty
```

### 12.4 Update knowledge

```python
reward = cfg.w_knowledge_update     * kg_quality_gain
       + cfg.w_uncertainty_reduction * uncertainty_reduction
       - action_costs["update_knowledge"]
       - redundancy_penalty
```

### 12.5 Stop (terminal, no redundancy)

```python
reward = cfg.w_stop_quality     * topk_match
       - cfg.w_stop_uncertainty * topk_uncertainty
       + stop_threshold_term
       - cfg.stop_cost

where
    stop_threshold_term = cfg.stop_success_bonus
        if topk_match >= cfg.stop_match_threshold
        else -cfg.stop_early_penalty
```

---

## 13. Coefficient defaults

| coefficient               |         default | purpose                                         |
| ------------------------- | --------------: | ----------------------------------------------- |
| `w_focus`                 |           `1.0` | focus-set Raman match improvement               |
| `w_pool`                  |           `0.5` | global candidate-pool improvement               |
| `info_gain_weight`        |           `0.5` | maximum information-gain contribution           |
| `info_gain_new_cap`       |            `10` | saturation cap for new candidates               |
| `info_gain_hits_cap`      |             `5` | saturation cap for high-value hits              |
| `high_value_similarity`   |          `0.80` | threshold for high-value candidate              |
| `diversity_weight`        |           `0.2` | structural diversity / novelty bonus            |
| `redundancy_penalty`      |           `0.1` | per-repeat penalty                              |
| `repeat_count_cap`        |             `5` | maximum repeat multiplier                       |
| `w_rank`                  |           `0.5` | reward weight for ranking improvement           |
| `rank_topk`               |             `5` | top-k candidates used for rank reward           |
| `w_rank_uncertainty`      |           `0.0` | optional penalty for uncertain top-k ranking    |
| `w_stop_quality`          |           `1.0` | final candidate quality reward                  |
| `w_stop_uncertainty`      |           `0.3` | penalty for stopping with uncertain candidates  |
| `stop_topk`               |             `5` | k for stop quality / uncertainty averages       |
| `stop_match_threshold`    |          `0.80` | minimum quality threshold for successful stop   |
| `stop_success_bonus`      |           `0.2` | bonus for stopping with good candidates         |
| `stop_early_penalty`      |           `0.2` | penalty for stopping too early                  |
| `stop_cost`               |           `0.0` | base cost of issuing a stop                     |
| `w_uncertainty_reduction` |           `0.3` | reward for reducing uncertainty                 |
| `w_knowledge_update`      |           `0.3` | reward for useful KG update                     |
| `kg_update_edge_cap`      |            `10` | saturation cap for updated KG edges             |
| `use_topk_delta`          |         `false` | use top-k mean instead of best-score delta      |
| `topk_delta_k`            |             `5` | k for top-k delta smoothing                     |
| `raman_hybrid_alpha`      |           `0.3` | full-spectrum weight in hybrid Raman similarity |
| `peak_sigma`              | domain-specific | Gaussian width for Raman peak matching (cm⁻¹)   |
| `peak_tolerance`          |     `3 * sigma` | peak matching tolerance                         |

Recommended action costs:

| action             | default cost |
| ------------------ | -----------: |
| `retrieve`         |       `0.02` |
| `expand`           |       `0.03` |
| `rank`             |       `0.01` |
| `query_dft_raman`  |       `0.20` |
| `update_knowledge` |       `0.01` |
| `stop`             |       `0.00` |

---

## 14. Expected reward magnitudes

With the default coefficients, typical per-action magnitudes are:

| term                      |                      typical range |
| ------------------------- | ---------------------------------: |
| `w_focus * delta_focus`   |                      `0.00 – 0.30` |
| `w_pool  * delta_pool`    |                      `0.00 – 0.15` |
| `information_gain`        |                      `0.00 – 0.50` |
| `diversity_novelty_bonus` |                      `0.00 – 0.20` |
| `rank_reward`             |                     `-0.20 – 0.30` |
| `DFT uncertainty reward`  |                      `0.00 – 0.30` |
| `action_cost`             |                      `0.00 – 0.20` |
| `redundancy_penalty`      | `0.00 – 0.50` (bounded by the cap) |
| stop quality term         |                      `0.00 – 1.00` |
| stop threshold adjustment |                     `-0.20 – 0.20` |

A useful exploration action usually falls around:

```
reward ≈ 0.2 to 0.8
```

A wasted or repeated action should be near zero or negative.

A successful stop is typically in `0.8 – 1.2`; a premature stop in `-0.2 – 0.4`.

---

## 15. Tuning procedure

Do not hand-tune all coefficients at once.

### Step 1 — Log reward components

For at least 100 held-out exploration episodes, log per action:

```python
{
  "action_type": ...,
  "delta_focus": ...,
  "delta_pool": ...,
  "information_gain": ...,
  "diversity_novelty_bonus": ...,
  "rank_delta": ...,
  "uncertainty_reduction": ...,
  "kg_quality_gain": ...,
  "action_cost": ...,
  "repeat_count": ...,
  "reward": ...,
}
```

### Step 2 — Check magnitude balance

Verify that no positive term contributes more than ~60% of `|reward|` on average.

If one term dominates, rescale **that term's coefficient**, not all of them.

### Step 3 — Tune only key coefficients

After magnitudes are calibrated, tune only a small subset:

```
w_focus
info_gain_weight
diversity_weight
query_dft_raman_cost
w_stop_quality
```

Optimize against a downstream metric such as:

* fraction of sessions that surface the ground-truth material in top-10
* average number of DFT calls before reaching a high-value candidate
* fraction of sessions that stop correctly (high `topk_match` at stop time)

---

## 16. Summary

This reward design evaluates exploration actions by asking:

1. Did the action improve Raman match quality?
2. Did it discover high-value candidates?
3. Did it explore a structurally diverse or novel region?
4. Did it reduce uncertainty?
5. Did it update the knowledge graph in a useful way?
6. Was the action worth its cost?
7. Was the action redundant?
8. Should the agent stop?

The agent is therefore trained not to modify materials, but to perform efficient Raman-informed structure exploration and knowledge feedback over a structure–Raman knowledge graph.
