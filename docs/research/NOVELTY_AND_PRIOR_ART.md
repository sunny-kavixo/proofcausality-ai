# Novelty and Prior-Art Positioning

This is a focused engineering positioning review, not a comprehensive patent or literature search. ProofCausality AI makes no global “first” claim.

## Relevant prior directions

| Direction | Representative reference | Relationship to ProofLoop |
|---|---|---|
| Active learning and query selection | Settles, *Active Learning Literature Survey* (2009), [Wisconsin technical report](https://minds.wisconsin.edu/handle/1793/60660) | Establishes selecting informative labels/queries; ProofLoop selects a concrete next visual observation for an evidence deficit. |
| Active visual recognition | Jayaraman & Grauman, *Learning to Look Around: Intelligently Exploring Unseen Environments for Unknown Tasks* (2018), [arXiv:1705.00119](https://arxiv.org/abs/1705.00119) | Studies view selection for recognition; ProofLoop is a verification policy with explicit abstention and receipt trace. |
| Selective classification / abstention | Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks* (2017), [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) | Provides risk-coverage framing; ProofLoop uses `INSUFFICIENT_EVIDENCE` and Evidence Debt to make the missing condition inspectable. |
| Predictive uncertainty | Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* (2017), [arXiv:1703.04977](https://arxiv.org/abs/1703.04977) | Motivates uncertainty-aware vision; ProofLoop does not claim calibrated Bayesian uncertainty. |
| Human-in-the-loop annotation | Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge* annotation workflows (2015), [IJCV](https://link.springer.com/article/10.1007/s11263-015-0816-y) | Demonstrates the importance of review workflows; ProofLoop adds a machine-readable reason and requested observation, but human effectiveness is untested. |

## Scoped implementation contribution

The implementation contribution is the composition of existing deterministic components into a portable evidence-closure contract:

1. a finite state trace from observation through re-verification;
2. explicit mapping from legacy action verdicts to claim semantics;
3. targeted recovery selection from measurable predicate deficits;
4. a follow-up endpoint that can remain unresolved rather than forcing success; and
5. a canonical receipt covering the transition and provenance boundary.

This is a system integration and auditability contribution. It is not a new segmentation architecture, trained foundation model, causal estimator, or proof that a physical intervention caused an outcome.

## Open validation questions

- Does the selected next observation improve resolution with real operators?
- Does the policy remain calibrated on independently labeled scenes and videos?
- How does it compare with learned active-perception policies under equal capture cost?
- Does temporal evidence improve action-completion decisions without being mistaken for causal attribution?
