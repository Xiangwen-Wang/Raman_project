# Knowledge Graph Schema — Raman-informed 2D Material Exploration Agent

This document defines the knowledge graph (KG) that the exploration agent navigates.

The agent is **read-only over structures**: it does not generate or modify materials. It only retrieves, expands, ranks, queries, updates, and explains candidate materials.

The KG is a **typed multigraph**:

* **Nodes** = materials, one node per database entry.
* **Edges** = typed relations between two materials.
* Each edge carries a `relation_type` and optional evidence attributes.
* Edges are **binary**: the graph only defines whether a relation exists.
* The KG does **not** assign hand-designed edge weights.
* Multiple edges of different types between the same pair are allowed and must be preserved.

The reason is that the relative importance of different relations is uncertain. Instead of manually deciding that one relation is stronger than another, the agent should learn which relation type is useful through its exploration policy and reward.

---

## 1. Node attributes

Every node stores at minimum:

| field                          | type             | notes                                                         |
| ------------------------------ | ---------------- | ------------------------------------------------------------- |
| `material_id`                  | str              | unique key                                                    |
| `formula`                      | str              | e.g. `MoSSe`                                                  |
| `prototype`                    | str              | structural prototype label, e.g. `2H`, `1T`, `1T'`            |
| `prototype_base`               | str              | normalized prototype label, e.g. stripping `-Janus` if needed |
| `family`                       | str              | coarse chemical family, e.g. `TMDC`, `MXene`                  |
| `metal`                        | str              | metal element symbol                                          |
| `chalcogen_top`                | str              | top-layer chalcogen                                           |
| `chalcogen_bottom`             | str              | bottom-layer chalcogen                                        |
| `is_janus`                     | bool             | `chalcogen_top != chalcogen_bottom`                           |
| `metal_atomic_mass`            | float            | atomic mass of metal element                                  |
| `chalcogen_top_atomic_mass`    | float            | atomic mass of top chalcogen                                  |
| `chalcogen_bottom_atomic_mass` | float            | atomic mass of bottom chalcogen                               |
| `mean_chalcogen_mass`          | float            | average mass of top and bottom chalcogens                     |
| `chalcogen_mass_asymmetry`     | float            | `abs(m_top - m_bottom)`                                       |
| `mass_vector`                  | np.ndarray[3]    | `[metal_mass, top_chalcogen_mass, bottom_chalcogen_mass]`     |
| `raman_spectrum`               | np.ndarray[4000] | raw / preprocessed Raman intensities                          |
| `raman_peaks`                  | list             | optional peak list for downstream explanation                 |
| `structure_embedding`          | np.ndarray[d_s]  | pretrained structure graph embedding                          |
| `raman_embedding`              | np.ndarray[d_r]  | pretrained Raman embedding                                    |

Derived fields such as `is_janus`, `prototype_base`, and atomic-mass features should be computed once at ingest time so the rules below can assume they exist.

Atomic mass is included because Raman peak shifts are often related to atomic mass changes, especially for chemically similar structures where heavier atoms tend to lower vibrational frequencies.

---

## 2. Design principles

Before listing the rules, the KG follows these conventions:

1. **No self-loops.**
   Never add an edge from a node to itself.

2. **Undirected edges.**
   Every rule below is symmetric. Store each pair once.

3. **Edges are unweighted.**
   A rule only decides whether a typed edge exists. It does not assign a scalar weight.

4. **Typed edges are preserved.**
   If two materials are related by multiple rules, keep all relation types. The set of relation types is part of the explanation.

5. **Edge attributes are evidence, not weights.**
   Values such as `composition_distance`, `shared_chalcogen_count`, `cosine_similarity`, or `atomic_mass_delta` can be stored as edge metadata, but they should not be treated as manually designed edge weights.

6. **The agent learns which edge type to follow.**
   The KG provides possible exploration routes. The exploration policy / action ranker learns whether to follow `same_prototype`, `atomic_mass_neighbor`, `raman_embedding_neighbor`, etc.

7. **Raw Raman spectra are kept on nodes, not used as raw cosine edges.**
   Raw Raman spectra and peak lists are used for visualization, Raman matching, and explanation. Graph-level Raman similarity is represented by `raman_embedding_neighbor`.

8. **Embedding-based edges are sparsified by percentile.**
   Absolute cosine thresholds may not transfer across datasets or embedding versions. For embedding relations, keep only the top `p%` most similar pairs.

---

## 3. Edge rules

The KG contains three groups of edge rules:

* **Group A:** symbolic / compositional relations
* **Group B:** atomic-mass relations
* **Group C:** embedding-based relations

All rules create edges without manual weights.

---

# Group A — Symbolic / compositional edges

These edges come from discrete material attributes. They are cheap, exact, and interpretable.

---

## R1. `structure_neighbor`

Same prototype and one atomic site away.

### Conditions

All must hold:

```python
prototype_base[a] == prototype_base[b]
composition_distance(a, b) == 1
```

where:

```python
composition_distance(a, b) =
    int(metal[a]            != metal[b])
  + int(chalcogen_top[a]    != chalcogen_top[b])
  + int(chalcogen_bottom[a] != chalcogen_bottom[b])
```

### Edge attributes

```python
relation_type = "structure_neighbor"
composition_distance = 1
changed_site = one of ["metal", "chalcogen_top", "chalcogen_bottom"]
is_janus_pair = (is_janus[a] != is_janus[b])
```

### Rationale

This edge captures materials that are structurally close and differ by one atomic substitution.

It does not mean the agent modifies one material into another. It only means these two materials are close neighbors in the existing structure space.

---

## R2. `same_prototype`

Same structural prototype.

### Conditions

```python
prototype_base[a] == prototype_base[b]
```

### Edge attributes

```python
relation_type = "same_prototype"
prototype_base = prototype_base[a]
```

### Rationale

This gives the agent a broad structure-based exploration route.

Unlike the previous weighted version, this edge is not suppressed when `structure_neighbor` also exists. If two materials are both `structure_neighbor` and `same_prototype`, both edges are preserved.

---

## R3. `shared_chalcogen`

The two materials share at least one chalcogen element.

### Conditions

```python
left  = {chalcogen_top[a], chalcogen_bottom[a]}
right = {chalcogen_top[b], chalcogen_bottom[b]}

shared = left & right
k = len(shared)

k >= 1
```

### Edge attributes

```python
relation_type = "shared_chalcogen"
shared_chalcogens = list(shared)
shared_chalcogen_count = k
```

### Rationale

This edge provides a chemically interpretable exploration route.

For non-Janus materials, the chalcogen set usually has size `1`.
For Janus materials, the set can have size `2`.

This edge is only a symbolic route. It does not imply that sharing a chalcogen is always more important than other relations.

---

## R4. `same_family`

The two materials belong to the same coarse chemical family.

### Conditions

```python
family[a] == family[b]
```

### Edge attributes

```python
relation_type = "same_family"
family = family[a]
```

### Rationale

This is a coarse exploration route. It is useful for broad search or cold-start graph construction, but the agent should learn whether this relation is useful in a given Raman-guided task.

---

## R5. `janus_analog`

Janus and non-Janus analogs with related composition.

### Conditions

All must hold:

```python
prototype_base[a] == prototype_base[b]
metal[a] == metal[b]
is_janus[a] != is_janus[b]
shared_chalcogen_count(a, b) >= 1
```

where:

```python
shared_chalcogen_count(a, b) =
    len(
        {chalcogen_top[a], chalcogen_bottom[a]}
      & {chalcogen_top[b], chalcogen_bottom[b]}
    )
```

### Edge attributes

```python
relation_type = "janus_analog"
is_janus_pair = True
shared_chalcogens = list(shared)
```

### Rationale

This edge is kept as an explicit relation type because Janus materials may introduce symmetry breaking and additional Raman-active modes.

Even if this relation overlaps with `structure_neighbor` or `same_prototype`, it should be preserved because it gives the agent an interpretable Janus-specific exploration route.

---

# Group B — Atomic-mass edges

Atomic mass is added as a separate exploration route because Raman frequencies are strongly related to vibrational masses and force constants. For structurally similar materials, replacing lighter atoms with heavier atoms often shifts Raman modes toward lower frequencies.

These edges do not claim that atomic mass alone determines Raman spectra. They only provide a physically meaningful route for exploration.

---

## R6. `atomic_mass_neighbor`

The two materials have similar atomic-mass patterns.

### Conditions

Compute the mass vector:

```python
mass_vector[i] = [
    metal_atomic_mass[i],
    chalcogen_top_atomic_mass[i],
    chalcogen_bottom_atomic_mass[i]
]
```

Compute normalized distance:

```python
mass_distance(a, b) =
    ||mass_vector[a] - mass_vector[b]||_2
    / mean(||mass_vector[a]||_2, ||mass_vector[b]||_2)
```

Create an edge if the pair is among the top `p_mass%` closest pairs by `mass_distance`, or if:

```python
mass_distance(a, b) <= cfg.atomic_mass_max_distance
```

Recommended defaults:

```yaml
kg:
  atomic_mass_top_percent: 2.0
  atomic_mass_max_distance: 0.15
```

### Edge attributes

```python
relation_type = "atomic_mass_neighbor"
mass_distance = mass_distance(a, b)
metal_mass_delta = abs(metal_atomic_mass[a] - metal_atomic_mass[b])
top_chalcogen_mass_delta = abs(chalcogen_top_atomic_mass[a] - chalcogen_top_atomic_mass[b])
bottom_chalcogen_mass_delta = abs(chalcogen_bottom_atomic_mass[a] - chalcogen_bottom_atomic_mass[b])
```

### Rationale

This edge lets the agent explore materials that are close in atomic-mass space, even if their symbolic labels are not identical.

It is useful for Raman-guided search because similar atomic masses may lead to similar vibrational frequency ranges.

---

## R7. `chalcogen_mass_trend`

The two materials differ mainly by chalcogen mass under a comparable structural setting.

### Conditions

All must hold:

```python
prototype_base[a] == prototype_base[b]
metal[a] == metal[b]
mean_chalcogen_mass[a] != mean_chalcogen_mass[b]
```

Optionally require:

```python
abs(mean_chalcogen_mass[a] - mean_chalcogen_mass[b])
    >= cfg.min_chalcogen_mass_delta
```

Recommended default:

```yaml
kg:
  min_chalcogen_mass_delta: 5.0
```

### Edge attributes

```python
relation_type = "chalcogen_mass_trend"
mean_chalcogen_mass_delta =
    mean_chalcogen_mass[b] - mean_chalcogen_mass[a]

lighter_material =
    material with smaller mean_chalcogen_mass

heavier_material =
    material with larger mean_chalcogen_mass

expected_raman_shift_direction =
    "heavier_chalcogen_likely_lower_frequency"
```

### Rationale

This edge captures a physically interpretable Raman trend:

```text
S → Se → Te
```

usually increases chalcogen mass and may shift related vibrational modes toward lower Raman frequencies, assuming comparable bonding and structure.

This relation is especially useful for TMD-like and Janus-TMD-like materials.

The edge is still undirected in storage, but the edge attributes record the lighter-to-heavier direction for explanation.

---

## R8. `janus_mass_asymmetry`

The two materials have related Janus mass-asymmetry patterns.

### Conditions

Both materials are Janus:

```python
is_janus[a] == True
is_janus[b] == True
```

and their chalcogen mass asymmetry is similar:

```python
abs(
    chalcogen_mass_asymmetry[a]
  - chalcogen_mass_asymmetry[b]
) <= cfg.janus_mass_asymmetry_tolerance
```

Recommended default:

```yaml
kg:
  janus_mass_asymmetry_tolerance: 5.0
```

### Edge attributes

```python
relation_type = "janus_mass_asymmetry"
mass_asymmetry_a = chalcogen_mass_asymmetry[a]
mass_asymmetry_b = chalcogen_mass_asymmetry[b]
mass_asymmetry_delta =
    abs(chalcogen_mass_asymmetry[a] - chalcogen_mass_asymmetry[b])
```

### Rationale

Janus materials break top-bottom symmetry. The mass difference between top and bottom chalcogen layers may be relevant to Raman-active modes and mode splitting.

This edge gives the agent a Janus-specific mass-based exploration route.

---

# Group C — Embedding-based edges

Embedding-based edges capture continuous similarities that categorical and mass rules cannot fully express.

They are still unweighted KG edges. Similarity values can be stored as metadata, but the agent should learn how much to rely on them.

---

## General recipe for embedding edges

For embedding vectors `e_a` and `e_b`:

1. L2-normalize once at ingest:

```python
e_hat = e / ||e||_2
```

2. Compute cosine similarity:

```python
sim = dot(e_hat_a, e_hat_b)
```

3. Sparsify by percentile:

```python
keep only top p% of unordered pairs
```

4. Create an edge if the pair is retained.

The cosine similarity is stored only as metadata:

```python
cosine_similarity = sim
```

It is not treated as a hand-designed edge weight.

---

## R9. `structure_embedding_neighbor`

### Conditions

The pair is among the top `p_structure_embedding%` most similar pairs by structure embedding cosine similarity.

Recommended default:

```yaml
kg:
  structure_embedding_top_percent: 1.0
  structure_embedding_min_sim: 0.5
```

### Edge attributes

```python
relation_type = "structure_embedding_neighbor"
structure_embedding_cosine = sim_structure
```

### Rationale

This edge captures structural similarity that symbolic rules may miss, such as geometrically or topologically similar structures with different prototype labels.

---

## R10. `raman_embedding_neighbor`

### Conditions

The pair is among the top `p_raman_embedding%` most similar pairs by Raman embedding cosine similarity.

Recommended default:

```yaml
kg:
  raman_embedding_top_percent: 1.0
  raman_embedding_min_sim: 0.5
```

### Edge attributes

```python
relation_type = "raman_embedding_neighbor"
raman_embedding_cosine = sim_raman
```

### Rationale

This edge captures spectroscopic similarity.

Raw 4000-dimensional Raman cosine is not used as a separate edge, because the Raman embedding already represents spectrum-level similarity. Raw spectra and peak lists remain available on the node for downstream Raman matching and explanation.

---

## 4. How the agent uses the KG

The KG does not produce a final hand-designed similarity score between two materials.

Instead, for a pair of nodes `(a, b)`, the graph returns:

```python
relations(a, b) = {
    edge_1.relation_type,
    edge_2.relation_type,
    ...
}
```

and the corresponding metadata:

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

The agent then learns which relation type is useful for a given exploration state.

For example, the agent may learn:

* follow `raman_embedding_neighbor` when the Raman target is specific
* follow `same_prototype` when searching within a structural family
* follow `chalcogen_mass_trend` when the target Raman peak should move to lower frequency
* follow `janus_analog` when extra Raman-active modes may be relevant
* follow `structure_embedding_neighbor` when symbolic labels are insufficient

The graph provides possible routes.
The action policy decides which route to take.

---

## 5. Example explanation

For example, the KG may return:

```text
MoS2 is related to MoSe2 through:
- same_prototype
- structure_neighbor
- shared_chalcogen
- chalcogen_mass_trend
```

The agent can explain:

```text
MoSe2 is explored from MoS2 because it has the same prototype and metal site,
but replaces S with the heavier Se chalcogen. This mass change provides a
physically meaningful Raman exploration direction, because heavier chalcogens
may shift related vibrational modes toward lower frequency.
```

For a Janus material:

```text
MoS2 is related to Janus MoSSe through:
- same_prototype
- structure_neighbor
- shared_chalcogen
- janus_analog
- chalcogen_mass_trend
```

The agent can explain:

```text
Janus MoSSe is explored because it is a Janus analog of MoS2, shares the same
prototype and metal, and introduces top-bottom chalcogen asymmetry. This may
change Raman activity through symmetry breaking and mass asymmetry.
```

---

## 6. Summary table

| Rule | Relation                       | Trigger                                                                   | Stored evidence                                           |
| ---- | ------------------------------ | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| R1   | `structure_neighbor`           | same prototype base and `composition_distance == 1`                       | changed site, composition distance, Janus-pair flag       |
| R2   | `same_prototype`               | same prototype base                                                       | prototype base                                            |
| R3   | `shared_chalcogen`             | at least one shared chalcogen                                             | shared chalcogens, shared count                           |
| R4   | `same_family`                  | same family                                                               | family                                                    |
| R5   | `janus_analog`                 | same prototype base, same metal, different Janus status, shared chalcogen | Janus-pair flag, shared chalcogens                        |
| R6   | `atomic_mass_neighbor`         | close in atomic-mass vector space                                         | mass distance, site-wise mass deltas                      |
| R7   | `chalcogen_mass_trend`         | same prototype base and metal, different mean chalcogen mass              | lighter/heavier direction, expected Raman shift direction |
| R8   | `janus_mass_asymmetry`         | both Janus and similar mass asymmetry                                     | asymmetry values and delta                                |
| R9   | `structure_embedding_neighbor` | top percentile by structure embedding cosine                              | structure embedding cosine                                |
| R10  | `raman_embedding_neighbor`     | top percentile by Raman embedding cosine                                  | Raman embedding cosine                                    |

