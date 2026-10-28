# Evidence Calculus Engine: Formal Typed Specification

**Engine Slug**: `EvidenceCalculusEngine`  
**Package**: `backend/evidence/calculus/`  
**Version**: `0.2.0`  

---

## 1. Mathematical Formalism

Let a visual verification task be defined by a tuple:
$$\mathcal{T} = (\mathcal{O}_{	ext{target}}, \mathcal{R}_{	ext{init}}, \mathcal{R}_{	ext{target}}, \mathcal{P}, 	au_{	ext{debt}})$$

Where:
- $\mathcal{O}_{	ext{target}}$ is the primary object category.
- $\mathcal{R}_{	ext{init}}, \mathcal{R}_{	ext{target}} \subset \mathbb{R}^2$ are the spatial reference regions.
- $\mathcal{P} = \{p_1, p_2, \dots, p_K\}$ is the set of observable predicates.
- $	au_{	ext{debt}} \ge 0$ is the maximum tolerable Evidence Debt (default $0.0$).

### Tri-State Semantics
Each predicate $p_k \in \mathcal{P}$ evaluates to an evidence state:
$$\sigma(p_k) \in \{	ext{SUPPORTED}, 	ext{CONTRADICTED}, 	ext{UNOBSERVABLE}\}$$

- $	ext{SUPPORTED}$: Unambiguous visual measurements satisfy the condition within calibrated perceptual bounds.
- $	ext{CONTRADICTED}$: Visible evidence explicitly violates the physical or spatial condition (e.g., placed outside target zone, physically impossible velocity jump).
- $	ext{UNOBSERVABLE}$: Evidence is missing, occluded, clipped by image borders, or corrupted by motion blur ($	ext{Var}(\Delta I) < 	heta_{	ext{blur}}$).

### Incompleteness vs Falsification Priority
The final verdict $V \in \{	ext{SUCCESS}, 	ext{FAILED}, 	ext{UNCERTAIN}\}$ follows strict epistemic priority:

$$V = egin{cases}
	ext{FAILED} & 	ext{if } \exists p_k 	ext{ s.t. } \sigma(p_k) = 	ext{CONTRADICTED} \
	ext{UNCERTAIN} & 	ext{else if } D(E, \mathcal{T}) > 	au_{	ext{debt}} \
	ext{SUCCESS} & 	ext{otherwise}
\end{cases}$$

> **Critical Axiom**: High model confidence can **never** override an `UNOBSERVABLE` or `CONTRADICTED` predicate.
