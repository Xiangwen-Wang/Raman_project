# Action Learning and Reward — Raman-informed 2D Material Exploration Agent

This document defines how the exploration agent learns which action to take in a Raman-informed 2D material knowledge graph.

The KG only defines whether typed relations exist between materials, such as:

* `same_prototype`
* `structure_neighbor`
* `shared_chalcogen`
* `janus_analog`
* `atomic_mass_neighbor`
* `chalcogen_mass_trend`
* `structure_embedding_neighbor`
* `raman_embedding_neighbor`

The agent must learn, from offline training episodes, which relation type or action is useful for finding the target material or high-Raman-match candidates.

It defines how to construct training episodes, how to evaluate exploration outcomes, and how to train an action policy / action ranker.

---

## 1. Core idea

The agent is trained to answer the following question:

Given a Raman target and the current exploration state, which action should the agent take next in order to find the most relevant material candidates efficiently?

The agent does not modify structures.

It only performs actions such as:

```text
retrieve
expand through a relation type
rank candidates
query DFT-Raman
update knowledge
stop
```

The KG provides possible routes.
The training data teaches the agent which routes are useful.

For example, the KG may tell the agent that from `MoS2`, it can explore:

```text
same_prototype
structure_neighbor
shared_chalcogen
chalcogen_mass_trend
raman_embedding_neighbor
```

But the KG does not decide which one is best.

The action policy learns this from offline exploration trajectories.

---

## 2. Relation between KG and action learning

The KG is an unweighted typed multigraph.

For two materials `(a, b)`, the KG returns:

```python
relations(a, b) = {
    "same_prototype",
    "structure_neighbor",
    "shared_chalcogen",
    ...
}
```

and optional evidence metadata:

```python
evidence(a, b) = {
    "composition_distance": ...,
    "shared_chalcogens": ...,
    "mass_distance": ...,
    "mean_chalcogen_mass_delta": ...,
    "structure_embedding_cosine": ...,
    "raman_embedding_cosine": ...
}
```

These values are **features**.

The agent may use them as input features, but it does not treat them as manually designed action scores.

The learned policy decides whether a relation is useful in the current Raman-guided exploration state.

---

## 3. Offline training environment

The offline training environment is built from the existing ideal 2D material structure–Raman database.

Each training episode is constructed by selecting one material or one Raman spectrum as the target.

For example:

```python
target_material = MoSe2
target_raman = Raman(MoSe2)
```

The agent receives the Raman target but does not directly receive the target material identity.

The task is to explore the KG and recover:

1. the ground-truth material, or
2. materials with high Raman similarity to the target.

This turns the existing database into an offline exploration environment.

---

## 4. State definition

At step `t`, the exploration state is:

```python
s_t = {
    "target_raman": y_star,
    "candidate_set": C_t,
    "visited_nodes": V_t,
    "frontier_nodes": F_t,
    "current_ranking": R_t,
    "relation_history": H_t,
    "remaining_budget": b_t,
    "available_actions": A_t
}
```

where:

* `target_raman` is the Raman objective or observed Raman spectrum.
* `candidate_set` contains currently discovered material candidates.
* `visited_nodes` are materials already explored.
* `frontier_nodes` are expandable materials.
* `current_ranking` is the current ranked candidate list.
* `relation_history` records which relation types have already been used.
* `remaining_budget` controls how many more actions or DFT calls are allowed.
* `available_actions` are valid actions from the current state.

The state is not a single material waiting to be modified.

It is:

```text
Raman target + explored candidate space + KG context + action history
```

---

## 5. Action space

The action space contains exploration operations, not structure modification operations.

---

### 5.1 `retrieve`

Retrieve initial candidates from the database.

Examples:

```python
{
    "type": "retrieve",
    "criterion": "raman_embedding"
}
```

```python
{
    "type": "retrieve",
    "criterion": "peak_window",
    "window_cm-1": [250, 320]
}
```

```python
{
    "type": "retrieve",
    "criterion": "prototype",
    "prototype": "2H"
}
```

Retrieve actions are used to create the initial candidate set.

---

### 5.2 `expand_by_relation`

Expand from the current candidates through a specific KG relation type.

Example:

```python
{
    "type": "expand_by_relation",
    "relation_type": "same_prototype"
}
```

Other possible relation types include:

```text
structure_neighbor
shared_chalcogen
same_family
janus_analog
atomic_mass_neighbor
chalcogen_mass_trend
janus_mass_asymmetry
structure_embedding_neighbor
raman_embedding_neighbor
```

This is the key action type.

The agent learns when each relation type is useful.

For example:

* `raman_embedding_neighbor` may be useful when the target spectrum is specific.
* `same_prototype` may be useful for local structural exploration.
* `chalcogen_mass_trend` may be useful when the Raman target suggests a lower-frequency shift.
* `janus_analog` may be useful when symmetry breaking or additional Raman modes are relevant.
* `atomic_mass_neighbor` may be useful when mass-driven vibrational similarity matters.

---

### 5.3 `rank_candidates`

Rank the current candidate set.

Example:

```python
{
    "type": "rank_candidates",
    "scoring_model": "learned_action_ranker"
}
```

Ranking is learned from training episodes.

It may use:

* Raman similarity features
* structure embedding features
* relation history
* atomic-mass features
* candidate uncertainty
* previous exploration outcomes

---

### 5.4 `query_dft_raman`

Query DFT-Raman for a selected candidate when Raman data is missing or uncertain.

Example:

```python
{
    "type": "query_dft_raman",
    "candidate_id": "Janus-MoSSe-1H"
}
```

This action is expensive and should be used selectively.

The agent learns when DFT feedback is worth the cost.

---

### 5.5 `update_knowledge`

Update the KG after new Raman data is obtained.

Example:

```python
{
    "type": "update_knowledge",
    "candidate_id": "Janus-MoSSe-1H",
    "new_raman": "DFT result"
}
```

The update may add or refresh:

* Raman spectrum
* Raman embedding
* Raman-neighbor edges
* structure–Raman evidence
* uncertainty estimates
* candidate ranking

---

### 5.6 `stop`

Stop exploration and return final candidates.

Example:

```python
{
    "type": "stop",
    "output": "top_k_candidates"
}
```

The agent should learn to stop when further exploration is unlikely to improve the result enough to justify the cost.

---

## 6. Raman match score

This signal is the Raman match between a candidate and the target.

The match score is not a KG edge weight.

It is used only for:

* constructing training labels
* computing offline rewards
* evaluating whether exploration succeeded
* ranking final candidates

Define:

```python
match_score(candidate, target) =
    raman_similarity(candidate.raman, target.raman)
```

The Raman similarity can be computed using:

* peak-level similarity
* full-spectrum similarity
* hybrid similarity

For example:

```python
match_score = raman_similarity(
    candidate,
    target,
    mode="hybrid",
    alpha=0.3
)
```

This score tells whether the agent found a good result.
It does not tell which KG edge is intrinsically stronger.

---

## 7. Training labels from offline exploration

For every training episode, the database already contains the answer.

The answer may be:

1. the exact target material, or
2. all candidates whose Raman match score exceeds a threshold.

Define:

```python
positive_candidates =
    {
        c in database
        if match_score(c, target) >= high_value_similarity
    }
```

Default:

```python
high_value_similarity = 0.80
```

If the task is exact identification, then:

```python
positive_candidates = {target_material}
```

If the task is Raman-guided discovery, then multiple high-match candidates may be valid.

The agent is trained to take actions that bring these positive candidates into the candidate set and rank them highly.

---

## 8. Offline trajectory construction

Each training trajectory is a sequence:

```python
(s_0, a_0, s_1, r_0),
(s_1, a_1, s_2, r_1),
...
(s_T, a_T, terminal_reward)
```

At each step:

1. the agent observes the current exploration state
2. it chooses an action
3. the environment updates the candidate set / frontier / ranking
4. the environment checks whether positive candidates were found
5. a reward or training label is assigned

The purpose is not to hand-design the best route.

The purpose is to generate training data so that the policy learns:

```python
Q(s, a) ≈ expected usefulness of action a in state s
```

where usefulness means:

```text
Does this action help find or rank the correct Raman-matching material?
```

---

## 9. Outcome-based reward

The reward is based on exploration outcome.

For non-terminal actions:

```python
reward =
    progress_reward
  + discovery_reward
  + ranking_reward
  + knowledge_reward
  - action_cost
  - redundancy_penalty
```

Each term is computed from what happened after the action.

No term uses a manually assigned KG edge weight.

---

## 10. Progress reward

Progress reward measures whether the action moved the agent closer to the Raman target.

Let:

```python
best_before = max match_score in candidate_set before action
best_after  = max match_score in candidate_set after action
```

Then:

```python
progress_reward =
    max(best_after - best_before, 0.0)
```

This rewards an action that discovers a candidate with better Raman match.

For pruning or filtering actions, negative progress should be allowed:

```python
progress_reward =
    best_after - best_before
```

because removing a good candidate should hurt.

---

## 11. Discovery reward

Discovery reward measures whether the action found new positive candidates.

Define:

```python
new_positive_hits =
    positive_candidates newly added to candidate_set
```

Then:

```python
discovery_reward =
    cfg.discovery_reward * min(
        len(new_positive_hits),
        cfg.discovery_hit_cap
    )
```

Recommended defaults:

```yaml
discovery_reward: 0.5
discovery_hit_cap: 3
```

This directly teaches the agent which action paths lead to the desired result.

For example, if expanding through `chalcogen_mass_trend` repeatedly brings high-match Raman candidates into the set, the agent will learn that this relation is useful in similar states.

---

## 12. Ranking reward

A good action should not only discover good candidates, but also place them near the top.

Let:

```python
topk_before = top-k ranked candidates before action
topk_after  = top-k ranked candidates after action
```

Define:

```python
topk_quality =
    mean match_score of top-k candidates
```

Then:

```python
ranking_reward =
    cfg.ranking_reward_weight
  * (topk_quality_after - topk_quality_before)
```

Do not clamp this value.

If ranking becomes worse, the reward should be negative.

Recommended defaults:

```yaml
ranking_reward_weight: 0.5
rank_topk: 5
```

This is especially important for the `rank_candidates` action.

---

## 13. Knowledge reward

For actions involving DFT or KG update, the agent can receive knowledge reward.

It is based on whether the action improves the training state.

Examples of useful knowledge updates:

* a missing Raman spectrum is filled
* a candidate uncertainty is reduced
* a new Raman embedding becomes available
* new typed edges are created
* the candidate ranking becomes more reliable

For v1:

```python
knowledge_reward =
    cfg.knowledge_reward_weight
  * uncertainty_reduction
```

where:

```python
uncertainty_reduction =
    max(uncertainty_before - uncertainty_after, 0.0)
```

Recommended default:

```yaml
knowledge_reward_weight: 0.3
```

For KG updates, optionally add:

```python
kg_update_reward =
    cfg.kg_update_reward_weight
  * min(number_of_new_edges / cfg.kg_update_cap, 1.0)
```

Recommended defaults:

```yaml
kg_update_reward_weight: 0.2
kg_update_cap: 10
```
---

## 14. Action cost

Each action has a cost.

Recommended defaults:

```yaml
action_costs:
  retrieve: 0.02
  expand_by_relation: 0.03
  rank_candidates: 0.01
  query_dft_raman: 0.20
  update_knowledge: 0.01
  stop: 0.00
```

DFT query has a higher cost because it is computationally expensive.

This teaches the agent not to use DFT unless the expected value is high.

---

## 15. Redundancy penalty

The agent should avoid repeatedly taking the same action in the same state.

Define:

```python
repeat_count =
    number of previous times the same action signature
    has appeared in the current episode
```

The action signature can be:

```python
(action_type, relation_type, source_node_or_candidate_set_hash)
```

Then:

```python
redundancy_penalty =
    cfg.redundancy_penalty
  * min(repeat_count, cfg.repeat_count_cap)
```

Recommended defaults:

```yaml
redundancy_penalty: 0.1
repeat_count_cap: 5
```

This discourages loops such as repeatedly expanding the same relation from the same candidates.

---

## 16. Terminal stop reward

The `stop` action is terminal.

It should be rewarded if the final result is good.

Define:

```python
final_topk = top-k candidates returned by the agent
```

Then:

```python
topk_match =
    mean match_score of final_topk
```

and:

```python
success =
    any candidate in final_topk belongs to positive_candidates
```

Terminal reward:

```python
stop_reward =
    cfg.stop_quality_weight * topk_match
  + cfg.stop_success_bonus  * int(success)
  - cfg.stop_failure_penalty * int(not success)
```

Recommended defaults:

```yaml
stop_quality_weight: 1.0
stop_success_bonus: 0.5
stop_failure_penalty: 0.5
stop_topk: 5
```

This teaches the agent when to stop.

Stopping is good if the target or high-match candidates have already been found and ranked highly.

Stopping too early is penalized.

---

## 17. Final reward by action type

### 17.1 Retrieve

```python
reward =
    progress_reward
  + discovery_reward
  + ranking_reward
  - action_costs["retrieve"]
  - redundancy_penalty
```

---

### 17.2 Expand by relation

```python
reward =
    progress_reward
  + discovery_reward
  + ranking_reward
  - action_costs["expand_by_relation"]
  - redundancy_penalty
```

The action includes a relation type:

```python
{
    "type": "expand_by_relation",
    "relation_type": "chalcogen_mass_trend"
}
```

The agent learns the value of the relation type from training outcomes.

---

### 17.3 Rank candidates

```python
reward =
    ranking_reward
  - action_costs["rank_candidates"]
  - redundancy_penalty
```

Ranking does not need to discover new candidates to be useful.

It is rewarded if it improves the top-k ordering.

---

### 17.4 Query DFT-Raman

```python
reward =
    progress_reward
  + discovery_reward
  + ranking_reward
  + knowledge_reward
  - action_costs["query_dft_raman"]
  - redundancy_penalty
```

DFT is useful only if the new Raman result improves candidate quality, reduces uncertainty, or helps future ranking.

---

### 17.5 Update knowledge

```python
reward =
    knowledge_reward
  + kg_update_reward
  - action_costs["update_knowledge"]
  - redundancy_penalty
```

An update that adds no useful information should receive near-zero or negative reward.

---

### 17.6 Stop

```python
reward = stop_reward
```

No redundancy penalty is applied because stop ends the episode.

---

## 18. Learning target

The final goal is to train an action policy or action ranker.

The model learns:

```python
Q(s, a) = expected future return after taking action a in state s
```

where:

```python
future return =
    discounted sum of rewards until stop
```

For offline training, each tuple is:

```python
{
    "state": s_t,
    "action": a_t,
    "next_state": s_{t+1},
    "reward": r_t,
    "terminal": done
}
```

The action ranker can then score candidate actions:

```python
score_action(s_t, a) -> expected utility
```

At deployment time, the agent chooses the action with the highest learned utility, not the largest KG edge weight.

---

## 19. What the policy should learn

The policy should learn patterns such as:

```text
When the target Raman peak is lower than the current candidates,
expanding through chalcogen_mass_trend may be useful.
```

```text
When the current candidates are structurally close but Raman mismatch remains high,
raman_embedding_neighbor may be more useful than same_family.
```

```text
When high-match candidates are already in the top-k,
stop may be better than additional expansion.
```

```text
When a candidate is high-potential but missing Raman data,
query_dft_raman may be worth the cost.
```

```text
When repeated expansion through the same relation gives no new positive candidates,
that relation should be avoided in similar states.
```

These behaviors are learned from training trajectories.

---

## 20. Coefficient defaults

| coefficient               | default | purpose                                          |
| ------------------------- | ------: | ------------------------------------------------ |
| `high_value_similarity`   |  `0.80` | threshold for positive Raman-match candidates    |
| `discovery_reward`        |   `0.5` | reward per newly discovered positive candidate   |
| `discovery_hit_cap`       |     `3` | cap for discovery reward                         |
| `ranking_reward_weight`   |   `0.5` | reward for improving top-k ranking               |
| `rank_topk`               |     `5` | top-k used for ranking reward                    |
| `knowledge_reward_weight` |   `0.3` | reward for reducing uncertainty                  |
| `kg_update_reward_weight` |   `0.2` | reward for useful KG updates                     |
| `kg_update_cap`           |    `10` | saturation cap for new KG evidence               |
| `redundancy_penalty`      |   `0.1` | penalty for repeated actions                     |
| `repeat_count_cap`        |     `5` | maximum repeat multiplier                        |
| `stop_quality_weight`     |   `1.0` | final top-k Raman quality                        |
| `stop_success_bonus`      |   `0.5` | bonus if final top-k contains positive candidate |
| `stop_failure_penalty`    |   `0.5` | penalty if stop fails                            |

Recommended action costs:

| action               |   cost |
| -------------------- | -----: |
| `retrieve`           | `0.02` |
| `expand_by_relation` | `0.03` |
| `rank_candidates`    | `0.01` |
| `query_dft_raman`    | `0.20` |
| `update_knowledge`   | `0.01` |
| `stop`               | `0.00` |

These coefficients define the offline learning signal.

---

## 21. Summary

This action-learning design matches the unweighted KG schema.

The KG defines:

```text
what exploration routes exist
```

The training set defines:

```text
which routes actually help find Raman-matching materials
```

The agent learns:

```text
which action to take under a given Raman target and exploration state
```
