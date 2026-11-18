# Probabilistic framework formalisation — Slice 7 one-stop theoretical reference

**Status.** Shipped with Slice 7 on 18 November 2026. This document is the
standalone theoretical reference accompanying `docs/architecture.md` Section 7,
intended for paper reviewers, academic collaborators, and anyone wanting the
math behind PinkMate's governance primitives.

**Canonical citations throughout:**
- Littman, M. L., Cassandra, A. R., & Kaelbling, L. P. (1995). *Learning policies for partially observable environments: scaling up.* In Machine Learning Proceedings 1995 (pp. 362–370). Morgan Kaufmann.
- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proceedings of the IEEE, 77(2), 257–286.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models.* Wiley Series in Probability and Statistics.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.

---

## 1. POMDP formalisation of the Q-MDP governance layer

### 1.1 The POMDP tuple

PinkMate's governance layer is a **Partially Observable Markov Decision
Process** `(S, A, T, O, R, γ)`:

```
S  = {MQL, SQL, ClosedWon, Disqualified}
A  = {score, nurture, convert}
Ω  = {engaged, not_engaged}
T(s, a, s') = P(s' | s, a)             transition kernel
O(s', a, o) = P(o | s', a)             observation kernel
R(s, a)     = expected immediate reward
γ ∈ [0, 1)  = discount factor          default γ = 0.95
```

The hidden state `s ∈ S` is the lead's true qualification level. The scorer
cannot observe `s` directly — only the touch-engagement observation
`o ∈ Ω`.

### 1.2 Bayesian belief filter

Given current belief `b: S → [0, 1]` (a probability distribution) and
executed action `a`, the Bayesian update after observing `o` is:

```
b'(s') = η · O(s', a, o) · Σ_{s ∈ S} T(s, a, s') · b(s)
```

where `η = 1 / P(o | b, a)` is the normalising constant. The marginal
`P(o | b, a)` is interpretable as the predictive likelihood of the
observation under the current belief.

**Implementation.** `pinkmate/qmdp_formal.belief_update(pomdp, belief, a, o)`
returns `b'` with `sum = 1` and raises `ValueError` if `P(o | b, a) = 0`
(the observation is impossible under the current belief — Bayes update is
degenerate).

### 1.3 Q-MDP commit rule (Littman et al. 1995, Eq. 4)

The Q-MDP policy approximates the optimal POMDP policy by weighting
*fully-observable* MDP Q-values with the current belief:

```
π_QMDP(b) = argmax_{a ∈ A} Σ_{s ∈ S} b(s) · Q_MDP(s, a)
```

where `Q_MDP(s, a)` is the optimal Q-function of the *underlying* fully
observable MDP, obtained by value iteration treating `s` as if observable.

Q-MDP is provably suboptimal vs the true POMDP optimal policy (which
requires PBVI, SARSOP or exact value iteration over the continuous belief
simplex), but it is:
- **tractable** (just solve the MDP, no belief-space planning), and
- **monotone in belief certainty** (higher `max_s b(s)` → more decisive
  action selection).

The second property is what makes Q-MDP the right governance primitive for
PinkMate: the system commits (fires the chosen action) when belief
concentrates, and defers (chooses conservative actions) when belief is
diffuse.

### 1.4 PinkMate's governance_commit as degenerate Q-MDP

PinkMate's Slice 1 `governance_commit(b, θ)` predicate is a **degenerate
Q-MDP** over the two-action space `A' = {commit, defer}`:

```
Q_MDP(s, commit) ∝ 𝟙[max_s b(s) ≥ θ]
Q_MDP(s, defer)  ∝ 1 - Q_MDP(s, commit)
```

So `π_QMDP(b)` on `A'` equals `commit` iff `max_s b(s) ≥ θ`, which is
exactly the predicate `governance_commit(b, θ)` exposed by
`pinkmate/qmdp.py::SuperposedBelief.governance_commit`.

This isomorphism is tested in `tests/test_qmdp_formal.py` via
`TestIsomorphismWithPinkMateSuperposedBelief`.

---

## 2. HMM interpretation of the touch sequence

### 2.1 Relationship to the POMDP

The POMDP with **no action dependence** on `T` and `O` reduces to a Hidden
Markov Model. Formally: an HMM is a POMDP with `|A| = 1` and `R ≡ 0` —
the agent has no choice, only observes the latent-state trajectory.

For PinkMate's downstream analytics (not the live governance loop), this
reduction is useful: given a batch of historical touch sequences, we can
estimate the latent-state trajectory of each lead without needing to
model the pipeline's action choices.

### 2.2 HMM primitives

Given discretised touch observations `o_1, ..., o_T ∈ {0, 1, ..., B-1}`,
the HMM is parameterised by:

```
π(s)       initial state distribution          shape (|S|,)
A(s, s')   transition matrix                   shape (|S|, |S|)
B(s, o)    emission matrix                     shape (|S|, B)
```

**Forward-backward algorithm.** Posterior over latent state at each step:
`γ_t(s) = P(s_t = s | o_1, ..., o_T)`. Marginalises over all compatible
latent paths.

**Viterbi algorithm.** Most-likely single latent path:
`s*_{1:T} = argmax P(s_1, ..., s_T | o_1, ..., o_T)`. Dynamic programming
in `O(T · |S|²)`.

**Baum-Welch EM.** Parameter estimation of `π, A, B` from unlabelled
observation sequences. Operationalised by `LatentStateHMM.fit()`.

### 2.3 Default PinkMate HMM

Latent states: `{MQL, SQL, ClosedWon}` (3 states, matches governance
state space minus the absorbing Disqualified).

Observation space: discretised touch counts `{0}, {1-3}, {4-10}, {11+}`
(4 bins).

Default transition matrix: slow left-to-right progression with self-loops
(`p_self = 0.7`, `p_forward = 0.3`), encoding the assumption that leads
progress monotonically through the funnel.

Default emission matrix: higher states emit more touches on average
(Gaussian-shape centred at `(s + 0.5) · (B-1) / |S|`).

---

## 3. Bayesian Weibull survival (Slice 7 extension of Slice 6)

### 3.1 Model

For time-to-conversion data `(t_i, e_i)` where `t_i ∈ ℝ_+` and `e_i ∈
{0, 1}` (0 = right-censored, 1 = observed event), the Bayesian Weibull
model is:

```
shape ~ HalfNormal(σ = 2.0)
scale ~ HalfNormal(σ = 2 · max_i t_i)

t_i | shape, scale, e_i = 1 ~ Weibull(shape, scale)        (observed)
P(T > t_i | shape, scale, e_i = 0)                         (censored)
    = exp(-(t_i / scale)^shape)                            S(t_i)
```

### 3.2 Posterior inference

NUTS (No U-Turn Sampler) with 2 chains × 1000 draws × 500 tune (defaults),
`target_accept = 0.9`. Convergence diagnosed by:
- **R-hat (Gelman-Rubin)**: < 1.01 desired, < 1.1 acceptable, > 1.1 indicates
  non-convergence.
- **Effective Sample Size (ESS)**: > 400 desired for reliable 95% CI estimation.

### 3.3 Posterior-predictive survival probabilities

Given posterior samples `(shape^{(m)}, scale^{(m)})_{m=1}^M`, the posterior
distribution of `S(t) = P(T > t)` at a specific time `t` is:

```
S^{(m)}(t) = exp(-(t / scale^{(m)})^{shape^{(m)}})
```

Equal-tailed 95% credible interval:
`[quantile_{0.025}(S^{(m)}(t)), quantile_{0.975}(S^{(m)}(t))]`.

This is **stronger** than Slice 6's point-estimate Weibull AFT because it
quantifies parameter uncertainty — for a regulatory evidence line, a
credible-interval statement ("P(convert by day 30) is 0.42 with 95% CI
[0.35, 0.49]") is more defensible than a point estimate alone.

---

## 4. Weibull mixture model for latent buyer regimes

### 4.1 Model

A `K`-component Weibull mixture with mixing weights `π_1, ..., π_K`
summing to 1:

```
f(t | θ) = Σ_{k=1}^K π_k · Weibull(t; shape_k, scale_k)
S(t | θ) = Σ_{k=1}^K π_k · exp(-(t / scale_k)^{shape_k})
```

Each component corresponds to a latent buyer regime: component 1 might
be fast converters (low scale, high shape), component 2 slow converters
(high scale, moderate shape).

### 4.2 EM algorithm with right-censoring

**E-step.** Responsibility of component `k` for observation `i`:
```
γ_{ik} = π_k · f_k(t_i)^{e_i} · S_k(t_i)^{1-e_i} / Σ_j [π_j · f_j(t_i)^{e_i} · S_j(t_i)^{1-e_i}]
```

**M-step.** Update mixing weights: `π_k = Σ_i γ_{ik} / N`.
Update per-component Weibull parameters via weighted MLE (Nelder-Mead on
the weighted log-likelihood including censoring contributions).

Convergence criterion: `|L^{(t+1)} - L^{(t)}| < 10^{-5}` where `L` is the
mixture log-likelihood. Default max 200 iterations.

### 4.3 BIC-based K selection (Decision C, 17 Nov 2026)

Fit the mixture at `K ∈ {2, 3, 4}`, compute BIC for each:

```
BIC(K) = -2 · L(K) + d(K) · log(N)
d(K) = 2K + (K - 1)        free parameters
```

Select `K* = argmin_K BIC(K)`. Lower BIC indicates better fit penalised
by model complexity.

---

## 5. Integration with the rest of the PinkMate architecture

### 5.1 Composition with Slice 2-6 modules

| This module | Composes with | Mechanism |
|---|---|---|
| `qmdp_formal.POMDP` | `pinkmate/qmdp.py::SuperposedBelief` | `pinkmate_superposed_belief_as_pomdp_belief()` adapter |
| `qmdp_formal.governance_commit` | `agents/scoring_agent.py` | theoretical reference for the Slice 1 commit predicate |
| `hmm_belief.LatentStateHMM` | `agents/enrichment_agent.py` touch features | optional post-hoc latent-state tagging for audit trails |
| `bayesian_survival.fit_bayesian_weibull` | `pinkmate/ml/survival.py::WeibullAFTFitter` (Slice 6) | credible-interval overlay on survival predictions |
| `mixture_survival.fit_weibull_mixture` | `pinkmate/ml/survival.py` | latent-regime segmentation on top of ensemble survival |

### 5.2 What this slice does NOT do

- **Does not change the production governance path.** `ScoringAgent` and
  `pipeline.py` continue to use the Slice 1 `qmdp.py` primitives. Slice 7
  is formal grounding, not a replacement.
- **Does not retrain Slice 6 survival models.** Bayesian Weibull and
  Weibull mixture are additional estimators alongside the Slice 6
  ensemble, not substitutes.
- **Does not rewrite the paper's Section III.A.** That's queued as part of
  the coordinated paper revisions pass (planned after Slice 7 lands). The
  module code and this document are the SOURCE MATERIAL for the paper
  rewrite; the rewrite itself is a separate deliverable.

---

## 6. Four locked decisions (recap)

| Decision | Choice | Rationale |
|---|---|---|
| A: Bayesian engine | PyMC 5.x+ | Maturity + Windows compatibility matter more than NumPyro's JAX speed on ~2000-row data |
| B: MCMC sampler | NUTS | Standard for Bayesian survival; R-hat + ESS diagnostics well-understood |
| C: Mixture K | BIC sweep K ∈ {2, 3, 4} | No upfront commitment to K=2; not overkill with Dirichlet Process |
| D: HMM library | hmmlearn ≥0.3, 3 latent states | Mature sklearn-style API; 3 states matches paper's belief-state terminology |

---

## 7. Test coverage

56 tests across 4 files:

- `tests/test_qmdp_formal.py` (16 tests) — POMDP construction, belief update known-answer (tiger POMDP), Q-MDP commit on dominant-SQL belief, governance commit threshold, isomorphism with `SuperposedBelief`.
- `tests/test_hmm_belief.py` (15 tests) — discretisation, HMM construction, Viterbi correctness, forward-backward row-sum invariant, log-likelihood finiteness.
- `tests/test_mixture_survival.py` (11 tests) — EM convergence on two-regime synthetic data (recovers distinct scales within factor-of-3), BIC-based K selection.
- `tests/test_bayesian_survival.py` (14 tests) — posterior recovery on true shape=2, scale=20 (95% CI covers truth), credible intervals monotone in level, posterior-predictive survival probability decreases with time, R-hat convergence.

Validated end-to-end locally at package build time (16.74s total runtime
including the PyMC NUTS fit).
