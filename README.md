# A Lorentzian CSS Duality in Causal Diamond Quantum Error-Correcting Codes
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19343889.svg)](https://doi.org/10.5281/zenodo.19343889) 
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![Doc License: CC BY 4.0](https://img.shields.io/badge/Doc_License-CC_BY_4.0-green.svg)](LICENSE-CC-BY.txt)


**Subtitle:** Four Codes from One Geometry via Orientation Reversal  
**Author:** Yannick Schmitt  
**Date:** March 2026  
**Status:** Preprint 1.0.0


## Overview

This paper shows that the discrete Lorentzian causal diamond `D` — the two-complex built on the twelve lightlike nearest-neighbour vectors of the ternary Minkowski lattice `{-1, 0, +1}^4` — generates not two but four CSS quantum error-correcting codes via a geometric duality, and that this duality makes the distance asymmetry `dZ = 2` of the original construction algebraically inevitable rather than merely observed.

The central mechanism is that the causal diamond supports two natural orientations of its boundary: the Lorentzian orientation, in which past links are incoming, produces the temporal charge `n^0_eff = 12` and motivates an all-ones X-check; the Euclidean orientation, in which all links are outgoing and the boundary sum vanishes, yields a dual code family in which the 21 plaquettes serve as X-type checks and the three spatial-axis groups serve as Z-type checks. These two orientations are related by Wick rotation `t → ix`, so the duality between the primal and dual code families is a quantum error-correction realisation of Wick rotation.

The four codes derived from this geometry are:

| Code | Parameters | Highlights |
|---|---|---|
| Code I | `[[12, 4, (4,2)]]` | Rate 1/3; corrects X-errors, detects Z-errors |
| Code II | `[[12, 1, (4,3)]]` | Balanced; circuit-level threshold `p_c ≈ 3.5%` |
| Dual A | `[[12, 2, (2,6)]]` | New; corrects all weight-1 and weight-2 Z-errors |
| Dual II | `[[12, 1, (3,4)]]` | New; `dZ = 4`, preferred for dephasing-dominated hardware |

Additional contributions include: a two-stage combined protocol that measures the 21 plaquettes alternately in Z- and X-basis to correct both error types simultaneously (`P_log = 0.006` at `p = 0.01` with `k_eff = 2`), a Pigeonhole No-Go theorem proving that `dZ ≥ 3` with `k ≥ 2` is impossible in the primal CSS family, and an X-Decoration Equivalence theorem extending this bound to weight-≤6 non-CSS codes.

## Repository Structure
* `/paper` - LaTeX source files and PDF pre-print of the manuscript.
* `/script` - Verification script

## Verification Script

All theorems, propositions, constructions, and numerical claims in the paper are verified by `verification_CSS_Duality_CD_QEC.py`. The script structure mirrors the paper sections exactly, and every check prints `PASS` or `FAIL` with its measured value.

### What the script verifies

| Section | Content |
|---|---|
| Sec 2 — Causal Diamond Geometry | Lightlike enumeration; 21 plaquettes; Laplacian spectrum; rank over R and GF(2) |
| Sec 3 — GF(2) Rank Gap | `rank_R(M) = 8`, `rank_{F2}(M) = 7`; all-ones vector as extra flat direction; 12 distinct weight-7 syndromes |
| Sec 4 — Lorentzian CSS Duality | Duality map `Φ`; isometric syndrome structure; primal ↔ dual distance exchange |
| Sec 5 — Code I `[[12,4,(4,2)]]` | Parameters; disjoint weight-4 X-stabiliser decomposition; code-capacity noise simulation |
| Sec 6 — Code II `[[12,1,(4,3)]]` | 1248 valid X-check sets; `dZ = 3` with 16 weight-3 Z-logicals; role of all-ones vector |
| Sec 7 — Dual A `[[12,2,(2,6)]]` | CSS commutativity proof; `dZ = 6`; 24 weight-6 Z-logicals; code-capacity performance |
| Sec 8 — Dual II `[[12,1,(3,4)]]` | Parameters; `dZ = 4` correction; `P_log(Z) = 0.001` at `p = 0.01` |
| Sec 9 — Two-Stage Protocol | Stage 1 X-correction and Stage 2 Z-correction; combined `P_log`; comparison table |
| Sec 10 — No-Go Theorems | Pigeonhole No-Go; X-Decoration Equivalence; exhaustive pair-killing counts |
| Sec 11 — Symmetry Group | Order-96 subgroup; rowspace preservation; duality symmetry |
| Sec 12 — Circuit-Level Thresholds | Code I and Code II circuit depths; `p_c ≈ 3.5%` for Code II |
| Miscellaneous | Additional checks not directly cited in the paper text |

### Requirements

```
numpy
```

### Running the script

```bash
python verification_CSS_Duality_CD_QEC.py
```

The script has a fast mode for a quick smoke-test (~2 minutes) and a full mode for complete paper-value verification. The most computationally expensive sections are the Monte Carlo noise simulations (Sec 9) and circuit-level threshold simulations (Sec 12). To enable fast mode, set `FAST = True` near the top of the script; this reduces trial counts from 20,000 to 2,000 for code-capacity simulations and from 10,000 to 500 for circuit-level simulations.


## Key Code Parameters

| Code | `n` | `k` | `dX` | `dZ` | Rate | Max stabiliser weight |
|---|---|---|---|---|---|---|
| Code I | 12 | 4 | 4 | 2 | 1/3 | 4 |
| Code II | 12 | 1 | 4 | 3 | 1/12 | 6 |
| Dual A | 12 | 2 | 2 | 6 | 1/6 | 4 |
| Dual II | 12 | 1 | 3 | 4 | 1/12 | 6 |

All four codes share the same 12-qubit physical layout; the qubit interaction graph is `K_12` (all-to-all connectivity), making neutral atom arrays and trapped-ion platforms the natural hardware targets.


## Series Context

- **Paper 1**: *Exact Discretisation and Boundary Observables in Lorentzian Causal Diamonds* — establishes the geometry, Laplacian spectrum, and BF crossover. Yannick Schmitt. (2026). Zenodo. https://doi.org/10.5281/zenodo.19338306
- **Paper 4**: *Algebraic Structure of the D4 Causal Diamond* — synthesises the geometry with symmetry group analysis, mass renormalisation, entanglement entropy, and MOND phenomenology. Yannick Schmitt. (2026). Zenodo. https://doi.org/10.5281/zenodo.19343721


## Citation

If you use this work, please cite it as:

> Yannick Schmitt. (2026). A Lorentzian CSS Duality in Causal Diamond Quantum Error-Correcting Codes. Zenodo. https://doi.org/10.5281/zenodo.19343889


## License
 * The source code in this repository is licensed under the [Apache License 2.0](LICENSE).
 * The documentation, LaTeX source files, and PDF papers are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE-CC-BY.txt).
