# Federated Semi-Supervised k-DPP

This repository contains stochastic local search with immutable forced sets
and a two-stage client-suggestion baseline. The current experiment deliberately
shares all client contexts with the server. Both final searches optimize joint
diversity with the same pooled 100-item context and 100 new items. This is a
non-private pooled-context baseline, not a solution to the private federated
research problem. Both improvement-only stochastic local search and DPPy
conditional k-DPP MCMC sampling are available. A separate three-party MPC
Euclidean SLS proof of concept is described below; the existing k-mer
experiments remain non-private and do not use that protocol.
The evolving semi-supervised problem remains in [writing/problem.tex](writing/problem.tex).

## Euclidean MPC proof of concept

For a visual walkthrough, open the self-contained
[interactive HTML explainer](supplements/mpc_euclidean_explainer.html) directly
in a browser. It covers secret sharing, protected arithmetic, cross-client
RBF entries, Schur conditioning, determinant geometry, and the SLS stopping
rule. Its teaching diagrams use invented data, separate from the measured
plaintext/MPC comparison; no private experiment fixtures are embedded.

Install the optional MPC dependencies and run the trusted synthetic evaluator:

```sh
python -m pip install -e '.[mpc,test]'
python experiments/run_mpc_euclidean.py --config configs/mpc_euclidean.yml
```

The configuration uses 24 uniformly distributed points in $[0,1]^4$. Six
points (25%) are randomly partitioned into three disjoint private contexts of
two points each; the remaining 18 points form the public candidate pool.
The RBF is $L_{xy}=\exp(-\gamma\lVert x-y\rVert_2^2)$ with $\gamma=1$ and
zero jitter. Unlike the k-mer Hamming kernel, Euclidean distance is squared.
All sizes, kernel parameters, precision, tolerances, trial counts, and stopping
settings are in [configs/mpc_euclidean.yml](configs/mpc_euclidean.yml).

Three independent MPyC processes construct the joint kernel, including
cross-client and private/public entries, without opening private coordinates.
They factor the protected private block and retain its protected Schur
complement over public candidates. Global SLS selects one shared $G_1$ of six
public points while keeping all six private points fixed. This is a numerical
and cryptographic arithmetic PoC; it optimizes logdet and does not sample an
exact k-DPP. Its pooled objective is one candidate approach, not a finalized
project method.

Patience is 50 proposals without an accepted logdet increase above $10^{-10}$.
The protected state freezes when patience is reached; the computation still
executes its fixed 200-proposal safety budget, concealing the stopping point.
If the cap is reached first, the result is labeled accordingly. Incoming ranks
are uniform over the eligible complement; outgoing slots are uniform over
$G_1$. Independent secret per-party coin shares prevent any one worker from
knowing the complete proposal tape. For each proposal the MPC computes
$\Delta=\log\det(L_{U\cup G_1'})-\log\det(L_{U\cup G_1})$ and the determinant
ratio $r=\exp(\Delta)$. This is not division of logdet values.

The separate trusted evaluator compares the full 24-point determinant,
initial/final joint logdet, all active swap ratios and decisions, stopping
counters, and final selection with an independent float64 reference using the
same data and proposal tape. Public results include a source hash, config,
versions, numerical errors, runtime, per-party peak memory, and measured
MPyC bytes. Actual communication rounds are not instrumented; explicit
barriers are reported separately.

MPyC provides Shamir secret sharing with three parties and threshold one:
at most one passive corrupt participant. This is MPC, not homomorphic
encryption. Only final public IDs and a numerical-validity bit are opened
unmasked. Diagnostic scores and histories use independent uniform field pads
from all three parties; the trusted evaluator can decode them, individual
workers cannot. Private inputs, coin shares, pads, and detailed audit
traces are kept in a separate mode-0700 `/tmp` directory printed by the
launcher, with mode-0600 files. Treat that directory as sensitive. Synthetic
validation scores in the report are deliberate diagnostic releases.

The trusted evaluator and a host administrator can access the synthetic
fixtures. This localhost benchmark uses loopback TCP without TLS and assumes
private authenticated channels; it is not a deployment or protection against
the host/network administrator. Peers connect through localhost, but MPyC
0.11's startup listener binding is not restricted by this launcher; run only
on an isolated trusted host/network. It hides coordinates rather than membership
in a fully public feature universe, so membership-private k-mer selection is
still a separate problem. No malicious security, differential privacy,
dropout recovery, exact sampling, or optimality is claimed. Secure polynomial
exp/log and fixed-point LDL arithmetic are approximate and must pass the
configured numerical checks without implicit jitter or clipping.

Sources: [MPyC security model](https://github.com/lschoe/mpyc),
[MPyC documentation](https://mpyc.readthedocs.io/en/latest/), and the expansion
paper's Section III Algorithm 1 and Equation (4). The code is in
`src/fed_skdpp/mpc_*.py`; the experiment script is a thin launcher.

## Run the experiments

From the repository root, install into a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dppy,test]'
```

The scripts are separate entry points to the same reusable package:

```sh
python experiments/run_numpy.py --config config.yml
python experiments/run_dppy.py --config config.yml
python -m pytest -q
```

You can also use the installed `fed-skdpp-numpy` and `fed-skdpp-dppy` commands.
Both scripts accept `--output-dir PATH`. Otherwise the output directory is
resolved relative to the configuration file. Each run creates a new directory;
previous results are not overwritten. If Matplotlib reports an unwritable
cache directory when importing DPPy, set `MPLCONFIGDIR` to a writable directory.

The NumPy-only installation is `python -m pip install -e '.[test]'`.
The optional DPPy dependency includes SciPy and Matplotlib.

## First experiment

The separate [Mermaid data-flow diagrams](supplements/centralized_federated_overlap.md)
show the parties, candidate pools, selection-comparison metrics, and plaintext
context disclosure in this baseline, alongside the distinct newer MPC
validation setup. They are not embedded in the HTML explainer.

Each trial samples 1,000 unique 6-mers uniformly without replacement from the
4,096 strings over `ACGT`. Reverse complements are distinct items. The same
public global collection is used by both approaches.

| Stage | Fixed context | Eligible additions | Selected additions |
|---|---|---|---|
| Centralized semi-supervised reference | The pooled 100-item $G_0$ | 900 global items outside $G_0$ | 100 items |
| Each of 10 clients | 10 items from its own 100-item partition | Remaining 90 local items | 50 suggestions |
| Server | The same pooled 100-item $G_0$ | 500 client suggestions | 100 public items |

Partitions cover the entire global collection. Because they are disjoint,
client suggestions cannot duplicate one another. Each $G_{0,i}$ is sampled
uniformly without replacement from its own partition using a separate seeded
RNG stream, and remains fixed throughout the trial. Client contexts therefore
do not overlap. Client suggestions exclude their own context and, by disjoint
partitioning, every other client's context too.

For the configured `pooled_context` baseline, each client uploads its 10
context IDs as well as its 50 suggestion IDs. Both final searches use exactly
the same $G_0=\bigcup_i G_{0,i}$ of size 100. The context is available for
scoring, but is fixed: its items cannot be selected again into $G_1$, swapped
out, or counted as new additions. This is the paper's no-overlap forced-set
interpretation, not an overlap experiment. Context membership is disclosed to
the server; this baseline makes no confidentiality claim.

Each participant uses the same kernel parameters and unit quality scores.
Item IDs refer to positions in
the public global collection; candidate-local indices are mapped back before
aggregation or evaluation.

The previous ordinary-DPP experiment (no forced sets, final size 50) is retained
in [configs/distributed_regular.yml](configs/distributed_regular.yml):

```sh
python experiments/run_numpy.py --config configs/distributed_regular.yml
```

The reusable code is in `src/fed_skdpp/`: configuration validation, data
generation, kernels, backend adapters, stochastic local search, metrics, and
experiment orchestration. `experiments/` contains no reusable algorithm code.

## Kernel and objective

For distinct strings $x,y$, the kernel is

$$
L_{xy}=\exp(-\gamma d_H(x,y))+\epsilon\mathbf{1}_{x=y},
$$

where $d_H$ counts mismatched positions, $\gamma>0$, and $\epsilon$ is configured
diagonal jitter. Hamming distance is not squared or normalized. The defaults
are $\gamma=0.5$ and $\epsilon=0$; gamma is illustrative, not a tuned choice.

Local client selection optimizes
$\log\det(L_{G_{0,i}\cup G_{1,i}})$, swapping only $G_{1,i}$. The local joint
set contains 60 items by default (10 fixed plus 50 suggested). After the server
chooses the shared 100-item $G_1$, each client's final augmented set contains
110 items. The fixed context's logdet is subtracted for conditional-gain
evaluation, but this constant does not change local swap acceptance.

With `selection.final_objective: pooled_context`, the centralized and server
searches both optimize

$$
U(G_1)=\log\det(L_{G_0\cup G_1}),
\qquad G_0=\bigcup_i G_{0,i},\quad |G_0|=100,\quad |G_1|=100.
$$

Their joint scoring sets contain 200 items. Conditional gains are
$U(G_1)-\log\det(L_{G_0})$; subtracting the same fixed context term changes
neither acceptance decisions nor the final utility gap. This pooled objective
is not a mean/minimum/other aggregation of individual client gains. It is
explicitly a non-private oracle/baseline and must not be presented as the
repository's private federated objective.

`selection.final_objective: regular` retains the previous centralized/server
$\log\det(L_{G_1})$ objectives and does not disclose context membership. Those
regular-diversity scores are not global semi-supervised utility.

All evaluations use float64 Cholesky log-determinants. Matrix shape,
finiteness, symmetry, and PSD
are checked. No implicit jitter or eigenvalue clipping is applied. A singular
or numerically unfactorable initial selection fails clearly; change gamma or
explicitly configure jitter rather than interpreting invalid utilities.

The RBF kernel is an extension of the expansion paper's Section IV-B,
Equation (6). Its original linear match-fraction kernel has rank at most 19
for DNA 6-mers and cannot give a nonzero determinant to these joint selections.

## Search and stopping

The shared optimizer adapts Section III, Algorithm 1 of
[Diverse Data Expansion with Semi-Supervised k-DPPs](<literature/Diverse_Data_Expansion_with_Semi-Supervised_k-Determinantal_Point_Processes.pdf>)
with an injectable objective. For a forced-set search:

1. Reserve the fixed $G_0$, then initialize a valid random $G_1$ of the declared
   new-item size, disjoint from $G_0$.
2. Choose one incoming item uniformly from outside $G_0\cup G_1$ and one
   outgoing item uniformly from $G_1$. Never remove an item from $G_0$.
3. Accept a swap only if the joint logdet does not decrease. Equal-scoring swaps are
   accepted, as in the paper.
4. Stop after 50 consecutive proposals with no accepted increase greater than
   the configured absolute `improvement_tolerance` (default $10^{-10}$).

Every evaluated proposal counts as an iteration, including rejections and
ties. An optional `max_iterations` is a safety cap; the default is `null`.
When the selected set is the entire candidate collection, there are no possible
swaps and the search returns immediately.

This stopping rule replaces the paper's fixed proposal budget. Fifty stalled
random proposals do not certify a local or global optimum. This is stochastic
local search, not the Metropolis chain described in Section 1 of
[Monte Carlo Markov Chain Algorithms](<literature/Monte Carlo Markov Chain Algorithms.pdf>).
Fixed-size DPP probabilities are defined in Section 3 of
[k-DPPs: Fixed-Size Determinantal Point Processes](<literature/k-DPPs Fixed-Size Determinantal Point Processes.pdf>);
the current optimizer does not sample from that distribution.

`semi_supervised_local_search` in `src/fed_skdpp/search.py` also provides the
single-context centralized paper baseline directly. With a public float64
kernel `L` and an integer array `g0` of 100 item indices:

```python
import numpy as np
from fed_skdpp.config import SearchConfig
from fed_skdpp.search import semi_supervised_local_search

result = semi_supervised_local_search(
    L, g0, selection_size=100,
    rng=np.random.default_rng(42), settings=SearchConfig(patience=50),
)
g1 = np.asarray(result.selected, dtype=np.int64)
# result.logdet scores L indexed by the 200-item union of g0 and g1.
```

Input context, optional initial selection, and returned selections use matrix
indices. Internally, the optimizer sees only eligible new-item positions.
Empty $G_0$ reduces to the regular objective. A singular context is rejected
because its conditional gain is undefined; jitter remains explicit.

### What the DPPy version does

For `search.algorithm: stochastic_local_search`, NumPy uses the L matrix directly.
The DPPy script wraps each stage's matrix in
`FiniteDPP("likelihood", L=L)` and supplies its L matrix to the same optimizer
and stable logdet routine. DPPy exposes exact and MCMC sampling, but no native
improvement-only optimizer with this patience rule. See the
[DPPy API](https://dppy.readthedocs.io/en/latest/finite_dpps/api.html).

Consequently, this is a comparison of kernel representations/integration, not
two independently implemented optimizers. Matching configs and seeds should
produce identical collections, partitions, selections, logdets, and Jaccard
values across scripts. Timings can differ, including DPPy import overhead.
The separate `dppy_mcmc` mode below actually runs DPPy's exchange sampler.

## Configuration and paired comparisons

All experimental settings live in [config.yml](config.yml):

- `data`: alphabet, `kmer_length`, and `global_collection_size`;
- `partition`: `num_clients`; collection size per client is derived;
- `selection`: `forced_set_size` per client, client suggestion size, final
  public selection size, and `final_objective` (`pooled_context` or `regular`);
- `kernel`: gamma and explicit jitter;
- `search`: algorithm, initialization policy, SLS patience/improvement tolerance/
  iteration cap, and MCMC `chain_steps`/`burn_in`;
- `numerics`: symmetry and PSD tolerances;
- `experiment`: root seed and number of repetitions;
- `output`: directory and whether to save search histories or full MCMC chains.

The global collection size must be divisible by the number of clients. Local
selection size plus forced-set size must not exceed a client collection; final
selection size must not exceed either the globally eligible collection or the
server's candidate pool. With a 100-item client pool and 10 fixed items, a client
cannot select 100 additional items; the 100-item size applies to the server's
shared final $G_1$, while each client still suggests 50.
Unknown configuration keys are errors. The client suggestion-size sweep below
reuses this configuration, varying only `selection.client_selection_size`.

The default `paired_uniform` initialization runs the client searches first,
then draws one random final-size set from the server pool. Centralized and
server searches both start with that set and receive identical proposal RNG
streams. Centralized proposals remain eligible over all global items outside
the private contexts. This eligibility collection is constructed only in the
centralized reference; the server's additions are restricted to client
suggestions even though its fixed context is disclosed. Thus centralized
**initialization** uses the compressed pool; its
search is not restricted to it. `independent_uniform` instead initializes
each final search independently within its own candidate collection.

Five trials use deterministic independent data, partition, initialization, and
client RNG streams derived from root seed 42. There is no hidden NumPy RNG.
Neither method is promised to win a finite search. Patience stopping does not
match compute budgets, so actual proposal counts and timings are reported.

### Client selection-size sweep

[configs/client_selection_sweep.yml](configs/client_selection_sweep.yml) lists
the sizes 10, 20, …, 90. A size of 100 is infeasible under the unchanged
1,000-item setup: each client reserves 10 of its 100 items as fixed context.

```sh
python experiments/run_client_selection_sweep.py --backend numpy
python experiments/run_client_selection_sweep.py --backend dppy
```

The sweep accepts `--config`, `--sweep-config`, and `--output-dir`. Matplotlib
is an optional dependency (`python -m pip install -e '.[plot]'`); it is already
available when the DPPy dependencies are installed.

Every parameter value runs the configured five paired trials with unchanged
data, partitions, contexts, kernel parameters, seed streams, and stopping rule.
For SLS, the plot's y-axis is the symmetric average nearest-neighbor Hamming
distance between the two final $G_1$ sets. Each selected sequence finds its
closest sequence in the other set; the two directed means receive equal weight.
Lower means more similar sequences, and identical sequences contribute zero.
The fixed $G_0$ is excluded. MCMC/exact-sampling comparison plots retain their
existing overlap diagnostics.
Blue points/error bars show mean ± sample standard deviation, and gray dots
show individual trials. These error bars are not confidence intervals.

Each unique sweep directory contains the full `results.json`, resolved base
and sweep YAML configs, per-trial `metrics.csv`, `hamming_distance_summary.csv`,
an SLS `report.md`, optional search histories, and PNG/SVG plots. The legacy
`overlap_summary.csv` is retained as a supplementary diagnostic. At size 90, no candidate compression
remains: both final searches have the same 900 eligible items and match under
paired initialization. At size 10, the server pool contains exactly the final
100 items, so the server has no alternative subset to swap into.

The centralized reference is re-run for each size. Its paired initialization
is drawn from that size's server pool, so it can change across the sweep;
this preserves the original experiment rather than introducing a different
fixed-reference comparison. This is still a non-private pooled-context
baseline with unequal actual compute budgets under patience stopping.

To repeat with five fresh trials, use the seed-43 configuration. The global
collection, client partitions, and fixed contexts differ between trials and
from seed 42; for each trial they remain identical across selection sizes.

```sh
python experiments/run_client_selection_sweep.py --config configs/client_selection_repeat.yml --backend numpy
python experiments/run_client_selection_sweep.py --config configs/client_selection_repeat.yml --backend dppy
```

### Local-pool retention

Analyze the existing saved sweep without re-running any searches:

```sh
python experiments/analyze_pool_retention.py --results results/PATH_TO_SWEEP/results.json
```

Let $P$ be the union of client suggestions, $A$ the full-universe centralized
final $G_1$, and $B$ the server's final $G_1$. The plot compares
$|P\cap A|$ with $|A\cap B|$, excluding every fixed context item. The first
curve counts centralized reference items still eligible after local selection;
$|A\setminus P|$ were excluded locally. The difference between curves counts
items available to the server but absent from its final selection. Because
$B\subseteq P$, overlapping the pool with the server final set itself always
gives the full final selection size and cannot diagnose candidate exclusion.

The analysis creates a new directory containing PNG/SVG plots, per-trial and
summary CSVs, and count-only JSON with source-file hash and original experiment
metadata. It preserves the original results. Error bars are sample standard
deviations. The centralized reference is a finite SLS result or terminal MCMC sample
whose initialization depends on the pool, not a unique optimum; membership loss does not establish
utility loss or biological significance.

### DPPy conditional k-DPP MCMC sweep

[configs/dppy_mcmc.yml](configs/dppy_mcmc.yml) keeps the same 1,000 DNA 6-mers,
10 disjoint clients, 10 forced items per client, pooled context of 100, final
public $G_1$ of 100, Hamming RBF gamma 0.5, zero jitter, and five seed-43 trials.
It reuses the previous seed-43 data, partitions and contexts, fixed across the
client suggestion sizes 10, 20, ..., 90. It changes the selection algorithm:
all clients and both final stages use DPPy's lazy exchange MCMC, not SLS.

```sh
python experiments/run_client_selection_sweep.py --config configs/dppy_mcmc.yml --backend dppy
```

For each stage, form the conditional likelihood kernel on eligible items $B$:

$$
L^{G_0}=L_{BB}-L_{BG_0}L_{G_0G_0}^{-1}L_{G_0B}.
$$

The implementation uses a Cholesky solve, not an explicit inverse. The identity
$\det(L_{G_0\cup Y})=\det(L_{G_0})\det(L^{G_0}_Y)$ means a fixed-size DPP on
this kernel is exactly the desired forced-set distribution. Singular contexts
fail explicitly. No jitter or eigenvalue clipping is added automatically; only
within-tolerance symmetry drift is corrected. The pooled server context is still
intentionally disclosed, so this remains a non-private baseline.

`src/fed_skdpp/mcmc.py` calls DPPy's actual
`dppy.mcmc_sampling.basis_exchange_sampler` implementation. Its proposal code
is unchanged: each transition is lazy with probability one half; otherwise it
proposes a uniform exchange and accepts with probability
$\min(1,\exp(\Delta\log\det))$. Decreases are allowed. Unlike SLS, this samples
sets with probability proportional to their joint determinant rather than
searching for a MAP optimum. A private copy of the native function's globals
replaces its raw-determinant helper with Cholesky log-weight arithmetic to avoid
underflow; the installed library, shared module globals, and proposal/acceptance
body are not modified. Tests compare its chain with the unadapted public DPPy
API on well-conditioned fixtures and frequencies with exhaustive tiny problems.
This numerical adapter is an explicit correction, not a change to the target
distribution; future DPPy changes require reviewing its helper contract.

`search.chain_steps: 10000` means 10,000 transitions, including lazy and rejected
steps, plus an initial state: saved chains have 10,001 rows. Every stage returns
the **terminal state**, never the best observed state. `burn_in: 2000` marks only
a diagnostic window for trace summaries; it deletes no states and changes no
terminal selection. Patience and the SLS iteration cap do not stop MCMC. A
single-state problem (e.g. selecting all 90 eligible local items) emits a constant
chain of the same requested length. The budget does not certify mixing;
low overlap between two samples is not itself sampling failure or utility loss.
At size 90, identical candidates, paired initialization and RNG streams make
the two final chains identical; they are coupled, not independent samples.

The command automatically saves overlap, logdet/gap/Jaccard and pool-retention
PNG/SVG plots, plus a representative final-stage chain trace. Metrics use
terminal selections and mean ± sample SD over the five paired trials. Proposal
counts exclude lazy transitions; equal chain budgets are not equal total compute
because the two-stage approach also runs clients. Full chains are streamed to
`chains/client_size_SIZE/trial_TRIAL/{centralized,server,client_ID}.npz`, avoiding
large JSON trajectories. Each file has `selected_ids` in public global index
space and `joint_logdet`; its absolute path and SHA256 are recorded in stage
metadata. `output.save_chain` controls these files, and `save_history` separately
controls JSON/CSV histories. No previous output directory is overwritten.

```python
import numpy as np

with np.load("PATH_TO_CHAIN.npz", allow_pickle=False) as saved:
    chain = saved["selected_ids"]       # (10001, selection_size), sorted unique IDs
    joint_logdet = saved["joint_logdet"]
    final_g1 = chain[-1]
```

For a writable Matplotlib cache and predictable reference timings, the run can
be prefixed with `MPLCONFIGDIR=/tmp/fed-skdpp-mpl OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1`. Thread settings are recorded in metadata. These settings
affect runtime, not the declared target distribution.

### Gamma sweep ranked by nearest-neighbor Hamming distance

The SLS configuration [configs/gamma_sls.yml](configs/gamma_sls.yml) retains the
original seed-42 experiment with 1,000 unique 6-mers, 10 clients, 10 fixed items
per client, pooled fixed context of 100, final public $G_1$ of 100, zero jitter,
and patience 50. The requested changes are local suggestion size 20 (a 200-item
server pool) and three trials per gamma. The explicit grid in
[configs/gamma_sweep.yml](configs/gamma_sweep.yml) is 0.05, 0.10, ..., 1.00.

```sh
python experiments/run_gamma_sweep.py --backend dppy
```

The command accepts `--config`, `--sweep-config`, `--backend`, and `--output-dir`.
Each gamma is used unchanged at all local clients, the centralized reference,
the server, and final evaluation. The same three datasets, partitions, fixed
contexts and RNG streams are reused across every value. Different trial IDs
have different datasets. Centralized/server initial selections match within
each trial and gamma; gamma-dependent client pools can change those initial
sets between gamma values. SLS and its stopping rule are unchanged, so compute
budgets can differ. The DPPy backend remains its likelihood-kernel representation
with the shared SLS optimizer; no exact or MCMC sampling is substituted.

Rank gamma by the **lowest** mean symmetric cross-set nearest-neighbor
Hamming distance over all three trials, excluding fixed context items.
Logdet, overlap, Jaccard and dispersion are not tie-breakers: exact equal mean
distances receive the same competition rank, and all co-winners are reported.
The mean ± sample SD plot shows individual trial distances and highlights
the lowest observed mean. No pool-retention
analysis is calculated. Each new directory contains full `results.json`,
resolved configuration and grid snapshots, per-trial `metrics.csv`,
`hamming_distance_summary.csv`, `gamma_ranking.csv`, `report.md`, optional SLS histories, and
`gamma_vs_nearest_hamming.png/.svg`. Legacy overlap summaries remain supplementary.
Existing results and default gamma are not
overwritten. Source/data hashes and numerical settings are retained.

This is exploratory tuning for agreement on these three synthetic trials,
not a universal best gamma, a guarantee of diversity, or a held-out performance
estimate. Larger gamma changes the kernel itself; do not rank gamma by raw
logdet or interpret values across gamma as one unchanged utility. This remains
the explicitly non-private pooled-context suggestion baseline.

### k-mer length sweep at fixed gamma

[configs/kmer_sls.yml](configs/kmer_sls.yml) fixes gamma at 0.25 for all local,
centralized, server and evaluation kernels. It keeps 1,000 unique DNA sequences,
10 disjoint clients, 10 fixed $G_{0,i}$ items per client, local suggestion size
20, final $G_1$ size 100, SLS patience 50, zero jitter, seed 42 and three trials.
Only `data.kmer_length` changes. The requested grid is 4, 6, ..., 20 in
[configs/kmer_sweep.yml](configs/kmer_sweep.yml). Length 4 is infeasible: there
are only 256 distinct DNA 4-mers, and even a smaller collection could not
provide 10 disjoint clients with 10 fixed plus 20 suggested items each. The
explicit `skip_infeasible: true` records that point without changing the setup;
the default API policy is to reject infeasible points before running any trial.

```sh
python experiments/run_kmer_sweep.py --backend dppy
```

The command accepts `--config`, `--sweep-config`, `--backend`, and `--output-dir`.
Each length/trial samples a fresh uniform collection of unique sequences.
Client partition IDs, fixed-context IDs and random streams stay paired across
lengths; their sequence values necessarily change. Within each point the two
final methods share an initialization, but length-dependent local pools can
change the initialization across points. DPPy supplies the likelihood-kernel
container; optimization is the unchanged shared SLS, not DPPy sampling or MCMC.
The patience rule does not equalize actual proposal counts.

Outputs include full results and configuration snapshots, Hamming-distance mean/sample
SD, per-trial metrics, optional search histories, explicit `skipped_points.csv`,
and PNG/SVG plots of raw and length-normalized nearest Hamming distance,
joint logdet, and paired logdet gap. The primary distance plot is
`kmer_length_nearest_hamming.png/.svg`; normalized distance divides by sequence
length and is supplementary. There is no pool-retention analysis. Each run uses a fresh directory.
This remains a non-private pooled-context baseline. Longer sequences change
raw Hamming distances and the kernel at fixed gamma, so compare the methods
within each length rather than interpreting raw logdet across lengths as a
shared utility scale. Reusing gamma-tuning seeds is exploratory sensitivity
analysis, not held-out validation. Root defaults and old results are preserved.

### SLS search-budget and convergence calibration

Run the centralized reference first, then each smaller client on the same
paired data, using the scientific settings in `configs/kmer_sls.yml` (6-mers,
gamma 0.25, three trials, centralized pooled $G_0$ of 100 and $G_1$ of 100;
each client has 100 items, $G_{0,i}$ of 10 and selects 20 new items):

```sh
python experiments/run_search_budget.py --backend dppy
```

All settings remain configurable with `--config`, `--budget-config`, `--backend`
and `--output-dir`. [configs/search_budget.yml](configs/search_budget.yml)
sets a 100,000-proposal cap, full swap audits every 1,000 proposals, and a
counterfactual reference patience of 50. Active patience stopping is disabled.
The expansion paper's Section III, Algorithm 1 uniform proposals and
nondecreasing acceptance are unchanged. Like its Section V-A, this studies
joint-logdet trajectories; the exhaustive checkpoint stopping rule is an
extension, not a replication or a global MAP guarantee. No audit-selected swap
is applied and audits do not consume stochastic proposal streams. The DPPy
backend is still a likelihood-kernel container, not DPPy sampling/MCMC.

Centralized initial selections are uniform over the full globally eligible
collection, independent of client preprocessing. Seed-paired collections,
partitions and contexts match previous experiments, but the central initializer
is intentionally no longer restricted to a client suggestion pool. Each local
search retains its original RNG/initialization stream. The first checkpoint
where no one-item swap improves logdet by more than `1e-10` is certified;
any uncertified run reaching the cap is reported as censored/incomplete, not
converged. The earliest checked certificate can lag actual stabilization.
An exhaustive audit uses float64 Cholesky/linear-solve identities for all pair
gains and verifies its best swap directly; no implicit kernel regularization
or clipping is used. Singular current kernels fail explicitly; singular
neighbors have logdet negative infinity and are not improving moves.

Each unique results directory has `results.json`, resolved `config.yml` and
`budget.yml`, `stages.csv`, `checkpoints.csv`, compressed NPZ trajectories,
and PNG/SVG logdet/convergence and remaining-swap-gain plots. NPZ arrays are
`logdet`, `accepted`, `improved`, and `stalled_iterations`, with index 0 the
initial state and subsequent indices counting attempted random swaps. Exact
initial/final/checkpoint public item IDs are saved separately in JSON; hashes
protect trace reproducibility. The trace prefix records when patience 50 would
have stopped and its utility loss relative to the measured terminal state.
Audited pair counts and audit/runtime costs are separate from proposal counts.
Curves are not extended beyond measured stopping times.

This is a **non-private optimizer study**, not a new federated protocol: no
suggestion upload, server reconciliation or network transport is executed.
Context membership and diagnostics are trusted research artifacts. Pooled G0
joint-logdet is not an aggregation of individual client gains. Different
central/client joint cardinalities mean their absolute scores are not directly
comparable; compare convergence budgets instead. Client measurements are
nested within three paired datasets, not 30 independent data trials. A tested
budget is sample-dependent, not a universal optimum. Root defaults, original
patience experiments and previous results are preserved.

## SLS with fixed local and global budgets

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_dppy.py --config configs/sls_fixed_budget.yml
```

This repeats the gamma-0.25, 6-mer SLS baseline on the same three seed-42 paired
datasets. Each client selects twenty new items after exactly 10,000 proposed
swaps. Centralized selection and server reconciliation each select 100 new
items after exactly 100,000 proposed swaps. The two settings are configurable:

```yaml
search:
  algorithm: stochastic_local_search
  local_proposal_budget: 10000
  global_proposal_budget: 100000
```

A non-null stage budget disables patience stopping for that stage and counts
every attempted swap, including rejection and ties. An incompatible smaller
`max_iterations` safety cap fails configuration validation. Empty/full
selections have no possible swap and report zero actual proposals. Accepted
moves, numerical tolerance, proposal generation, and explicit RNG streams are
unchanged. Fixed proposal budgets recover the expansion paper's Section III,
Algorithm 1 stopping rule; they do not establish a MAP or convergence guarantee.

Ten disjoint 100-item clients each reserve ten immutable context items. The
server pool contains the 200 local suggestions. Both final stages condition
on the same disclosed pooled G0 of 100 and optimize joint logdet. This is the
existing **non-private pooled-context baseline**. The shared final starting
set is drawn from the current server pool, using the original paired-uniform
policy and common final proposal stream. Longer local search can change that
pool and therefore change the shared starting set relative to a patience run.
Dataset, partition and context pairing is preserved; initialization values
are recorded explicitly.

Fresh `dppy_sls_fixed_budget_*` directories contain selections, counters,
initial/final utilities, directed and symmetric nearest-neighbor Hamming distances,
normalized distance, configuration, source
metadata, and `sls_fixed_budget_comparison.png/.svg`. The provided configuration
omits large histories; `output.save_history: true` restores complete
trajectories. The DPPy backend holds L while the shared SLS implementation runs
the stochastic search. Prior SLS, exact-sampling and MCMC artifacts are retained.

## DPPy exact Kulesza–Taskar sampling

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_dppy.py --config configs/dppy_exact.yml
```

This repeats the centralized/two-stage comparison with one exact conditional
k-DPP draw at every client and at each final stage. The configuration uses the
same seed-42 paired datasets as the gamma-0.25 SLS experiments: three trials,
1,000 unique 6-mers, ten disjoint 100-item client partitions, ten fixed context
items per client, twenty local suggestions, and a final G1 of 100. Each trial
has a 200-item server suggestion pool and a shared 100-item pooled context.
This is the existing **non-private pooled-context baseline**; the server sees
all client contexts, and pooled joint logdet is not an aggregation of client
conditional gains. Participation is complete, with no noise or secure
aggregation. The existing communication accounting and trusted evaluation apply.

`search.algorithm: dppy_exact` dispatches to `src/fed_skdpp/exact.py`, which calls
`FiniteDPP.sample_exact_k_dpp(size=selection_size, mode="KuTa12", ...)`.
Kulesza and Taskar (2011), Section 3.1 Algorithm 2, first chooses eigenvectors
using elementary symmetric polynomials, then invokes the projection-DPP sampler
from Algorithm 1. DPPy implements this with the positive ESP recurrence and
the `KuTa12` projection sampler from the 2012 survey. `search.exact_mode` also
accepts `GS` and `GS_bis`; the experiment uses `KuTa12`. See the
[DPPy exact-sampling documentation](https://dppy.readthedocs.io/en/latest/finite_dpps/exact_sampling.html).

At each stage, the immutable G0 is conditioned out with a Cholesky solve,
using the same Schur-complement helper as the MCMC path. The sampled size is
the number of **new items**, not the size of G0 union G1. The native library
receives the validated float64 eigendecomposition and its own ESP table.
Singular contexts, negative computed eigenvalues, insufficient numerical rank,
invalid ESP normalizers, or disagreement between conditional and direct joint
logdet fail explicitly. No automatic jitter, clipping, truncation, or kernel
rescaling is applied. The DPPy numerical rank threshold is recorded; the full
eigenspectrum is supplied to the sampler.

There is no initial selection, chain, patience stopping, or optimization
trajectory. SLS/MCMC budget fields and `initialization` are unused for exact
sampling. Each stage returns its single draw, with zero swap proposals and an
explicit sample count. Central and server sampling always use independent RNG
streams (3 and 4). Their overlap is measured only on G1, excluding the common
G0. Even with no client compression, independent exact draws can differ. In a
one-client, full-pool test, both final stages recover the same conditional
distribution; a shared RNG reproduces the same realization.

Fresh `dppy_exact_*` result directories include resolved config, dependency and
source metadata, per-trial metrics, every selected set, sampler seeds,
normalizers, spectral diagnostics, runtimes, and `exact_comparison.png/.svg`.
Reported means and sample standard deviations summarize the paired data trials;
three trials give only preliminary evidence. The exact sampler draws from the
specified distribution rather than maximizing logdet, so membership overlap
and utility need to be interpreted separately. The two-stage construction is
not asserted to reproduce the centralized k-DPP distribution.

## Ordinary local DPPs with an exact server size sweep

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fed-skdpp-mpl python experiments/run_variable_pool_baseline.py --config configs/variable_pool_baseline.yml
```

[configs/variable_pool_baseline.yml](configs/variable_pool_baseline.yml) defines
ten clients with 1,000 unique 20-mers each, the Hamming RBF
$L_{xy}=\exp(-0.6d_H(x,y))$, zero jitter, seed 42, and three fresh data trials.
All data, kernel, client-count and size-grid parameters are configurable.
Each client draws an **ordinary variable-size DPP**, using native DPPy
`sample_exact(mode="GS")`; there is no local selection-size parameter.
The merged pool has random size $m$, the sum of the local draw sizes.

The server draws a size-$k$ **exact k-DPP on the merged pool** for
$k=10,15,\ldots\leq m$, appending $m$ when it misses the grid. There is one
server draw per size per trial. The data, partitions, and local pool remain
fixed across sizes within a trial; all are refreshed between trials. Rare
pools smaller than ten yield only their full-pool endpoint, without redrawing.

The server uses a separate exact rejection implementation, **not DPPy's
fixed-size spectral API**. Hadamard bounds uniform-subset rejection weights;
Jacobi complementary minors reduce the work for sizes larger than half the
pool. Shared proposal permutations, thresholds, and prefix Cholesky factors
allow many sizes to be sampled together without overflowing ESP tables.
Each size has the correct marginal k-DPP law, but draws across sizes are
correlated and not necessarily nested. No approximate kernel, eigenvalue
clipping, SLS or MCMC is substituted. The derivation, floating-point checks,
and exact-law enumeration/frequency tests are in
[variable_dpp.py](src/fed_skdpp/variable_dpp.py) and
[test_variable_pool.py](tests/test_variable_pool.py).

The global curve reports $\log\det(L_Y)$. At each size the boxplot has ten
client scores: $\log\det(L_{Y\cap S_i})$, where $S_i$ is the client's original
partition. Empty contributions have score zero. Counts are saved and shown
in a companion heatmap, since logdet comparisons across different sizes are
not size-controlled diversity comparisons. Another boxplot shows each
client's initial ordinary-DPP logdet and size across the three trials.

This is a **context-free, non-private aggregation baseline**, with all private
contexts empty and selected IDs shared in plaintext. It is not a proposed
private semi-supervised protocol or a claim that two-stage samples follow
the full-10,000-item k-DPP law. All clients participate; no failures or non-IID
partitioning are modeled. Dense kernels cover clients and the realized pool
only; no full $10{,}000\times10{,}000$ kernel is built.

Every `variable_pool_dpp_baseline_*` directory contains a Markdown report,
resolved YAML, public data/partitions, lossless compact representations of all
server subsets, numerical and environment metadata, runtime/peak RSS,
logical int64-ID communication counts, CSV scores/counts, and four figures in
PNG/SVG: `server_logdet`, `client_logdet_boxplots_by_size`,
`client_contribution_heatmaps`, and `local_client_boxplots`.
`server_summary.csv` includes the number of trials contributing at each size
because the random pool endpoints differ. JSON/CSV client IDs are zero-based;
figure labels are one-based.

### Centralized sweep on the same saved 10,000-item datasets

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/fed-skdpp-mpl python experiments/run_centralized_pool_baseline.py --config configs/centralized_pool_baseline.yml
```

[configs/centralized_pool_baseline.yml](configs/centralized_pool_baseline.yml)
points to the distributed baseline's saved `results.json`. All three original
10,000-item datasets and their exact row/global-ID order are reused; collection
and source-file SHA-256 hashes are checked and recorded. No data, partitions,
or local DPP suggestions are regenerated. The alphabet, 20-mer length,
Hamming RBF gamma 0.6, zero jitter and numerical tolerances are inherited.

Draw one exact k-DPP from the **full global collection** at sizes
$10,15,\ldots,5000$: 999 sizes per source trial, with independent centralized
RNG streams. Minimum, maximum, interval and sampling seed are configurable.
The tested rejection sampler is shared with the distributed baseline, not
DPPy's fixed-size spectral API. Across-size draws are coupled, rather than
independent or necessarily nested. This is a non-private, context-free
centralized reference, not a logdet optimizer or a private federated protocol.
It deliberately uses a dense $10{,}000\times10{,}000$ kernel, unlike the
distributed baseline's client/pool kernels.

Fresh `centralized_global_pool_kdpp_*` directories contain every sample in
lossless compact form, the original public datasets, source provenance,
resolved/inherited configs, sampler and numerical checks, runtime/peak RSS,
`centralized_logdet.csv`, `centralized_summary.csv`, and PNG/SVG plots.
`comparison.csv` and `comparison_summary.csv` additionally compare the
centralized draws with the saved distributed draws **at common sizes only**;
positive gaps mean larger centralized logdet. Distributed pools below 5,000
have no size-5,000 comparison, so paired trial counts are reported rather
than filling in missing scores. These are paired datasets, not matched sample
IDs. Neither exact sampler guarantees an optimal score or higher utility than
an independently drawn sample.

### Global-to-selected nearest-neighbor coverage boxplot

For the requested comparison between all global items and the selected subset,
run `experiments/analyze_subset_nearest_neighbors.py --centralized-results PATH_TO_RESULTS_JSON --selection-size 1000 --metric global_to_subset`
with the project Python environment. Global-to-selected coverage is the CLI
default. For each of the 10,000 global items, find its nearest selected item
and average those raw Hamming distances. Selected global items contribute zero.
**Lower is better coverage.** No diagonal masking or exclusion of selected
global items is applied. This differs from within-subset separation below.

Each box contains the three paired trial averages, not 30,000 independent
observations. Fresh `global_to_subset_nn_*` directories contain PNG/SVG plots,
trial means, all per-global-item distances, input hashes and a report. No
source samples are rerun or overwritten. This corrects the earlier
within-subset plot for the global-versus-selected comparison.

For three per-trial panels with both methods' **per-global-item** distance
distributions, run `experiments/plot_global_subset_distance_trials.py --analysis-results PATH_TO_COVERAGE_RESULTS_JSON`.
Use a saved `global_to_subset_nn_*` analysis, not a centralized sampling file.
Each panel has two boxes containing all 10,000 distances. Whiskers mark the
actual minimum/maximum, black lines mark medians, and red diamonds mark means.
Numeric min/max/mean/median tables, a CSV, and separate images for each trial
are saved to a fresh `global_to_subset_nn_trial_boxes_*` directory. No samples
or distances are recomputed, and earlier figures remain unchanged.

### Within-subset nearest-neighbor separation (different diagnostic)

To compare existing size-1,000 subsets without rerunning sampling, run
`experiments/analyze_subset_nearest_neighbors.py --centralized-results PATH_TO_RESULTS_JSON --selection-size 1000 --metric within_subset`
with the project Python environment. The distributed result is resolved from
the centralized run's recorded source. Input hashes and paired datasets are
verified; output goes into a fresh `within_subset_nn_*` directory.

This analysis averages each selected item's Hamming distance to its nearest
**other selected item**. Higher values mean greater local separation. It is
not full-dataset-to-subset coverage, for which lower distance is better.
The boxplot contains one average per paired data trial (three points per
method for the saved baseline), with paired points connected. Per-item
distances are descriptive data, not independent experimental replicates.
The directory includes PNG/SVG plots, trial means, per-item CSV, provenance,
summaries, and a report. No original results or optimizer metrics are changed.

For **per-selected-item** boxplots comparing centralized and federated subsets
in each trial, run
`experiments/plot_within_subset_distance_trials.py --analysis-results PATH_TO_WITHIN_SUBSET_RESULTS_JSON`.
This reads a saved `within_subset_nn_*` analysis and does not rerun sampling
or distance calculations. Each method's box contains its selected items'
nearest-other Hamming distances, with self-matches excluded. Markers and
numeric tables show the mean, median, 5th percentile and 10th percentile;
whiskers show the actual min/max. Higher values indicate greater separation
at the same subset size. Percentiles use linear interpolation. A fresh
`within_subset_nn_percentiles_*` directory contains combined and individual
trial PNG/SVG plots, distance statistics, paired centralized-minus-federated
differences, provenance and a report. These boxes describe dependent item
distances within each trial, not independent experimental replications.

## Metrics and result files

### Primary SLS selection comparison

SLS still optimizes its declared logdet objective. The primary **evaluation**
comparison between centralized $A=G_{1,\mathrm{central}}$ and federated
$B=G_{1,\mathrm{fed}}$ is now

$$
D_{\mathrm{NN}}(A,B)=\frac12\left[
\frac1{|A|}\sum_{a\in A}\min_{b\in B}d_H(a,b)
+\frac1{|B|}\sum_{b\in B}\min_{a\in A}d_H(a,b)
\right].
$$

This uses raw mismatch counts, not squared distances or RBF similarities.
Nearest matches are many-to-one, not an optimal one-to-one assignment.
Exact matches contribute zero; fixed context sets are excluded. Lower means
more similar final selections, **not** better diversity, higher logdet, or a
privacy guarantee. This symmetric nearest-neighbor dissimilarity need not
satisfy the triangle inequality.

New SLS trial metrics include:

- `centralized_to_federated_mean_nearest_hamming`;
- `federated_to_centralized_mean_nearest_hamming`;
- `symmetric_mean_nearest_hamming` (primary, range zero to k-mer length);
- `symmetric_mean_nearest_hamming_normalized` (distance divided by length).

To regenerate reports, plots and gamma rankings from saved SLS selections,
without rerunning any searches:

```sh
python experiments/reanalyze_sls_hamming.py --all-saved
python experiments/reanalyze_sls_hamming.py --results results/PATH_TO_RUN/results.json
```

The first command scans `results/*/results.json`. Optional arguments are
`--results-root`, `--output-dir`, and `--no-plots`. It creates a fresh
`sls_hamming_reanalysis_*` directory with an index report, source-file hashes,
per-run compact results, CSVs and plots. Original files are never overwritten;
saved selections, kernels, logdet scores, budgets and runtimes do not change.
Large histories are not duplicated. MCMC, exact sampling, Euclidean MPC and
central-only search-budget calibration records are explicitly skipped: they
are outside this paired k-mer SLS change.

Overlap, Jaccard, and local-pool retention remain distinct supplementary
diagnostics; neither overlap nor Jaccard ranks gamma in SLS anymore. Historical
plots and reports are retained as records of their original evaluation.

### Logdet utilities and supplementary diagnostics

With pooled-context scoring, both final outputs use the same context, kernel
parameters, jitter, and joint cardinality:

$$
\Delta=U(G_{1,\mathrm{fed}})-U(G_{1,\mathrm{central}}),
\qquad
J=\frac{|G_{1,\mathrm{central}}\cap G_{1,\mathrm{fed}}|}
{|G_{1,\mathrm{central}}\cup G_{1,\mathrm{fed}}|}.
$$

Higher joint logdet is better; positive $\Delta$ favors the distributed result
on this non-private pooled-context objective. The `centralized_logdet` and
`federated_logdet` metrics now report these joint scores, not $G_1$-only scores.
Separate `*_regular_logdet` metrics preserve ordinary diversity diagnostics;
`*_conditional_gain` subtracts the shared context logdet. Jaccard compares
only the new $G_1$ selections, not their unions with the common context.

In addition, `trusted_evaluation_client_scores` records
each client's joint logdet and conditional gain for both final sets:

$$
g_i(G_1)=\log\det(L_{G_{0,i}\cup G_1})-\log\det(L_{G_{0,i}}).
$$

These per-client diagnostics are not the pooled-context objective and are not
aggregated into a global objective. They and local search histories are
trusted, non-private evaluation artifacts,
not server-visible messages. Do not share them as privacy-safe logs: exact
scores and suggestions may leak information about private membership.

In `pooled_context` mode, intentionally disclosed context IDs are serialized
as `server_shared_context_ids` and `client_disclosed_context_ids`. In `regular`
mode, context IDs are not serialized, but the saved configuration and root seed
still allow a trusted evaluator to reconstruct synthetic contexts. Publishing
those evaluation files does not provide confidentiality in either mode.

Jaccard measures identical item membership, not sequence resemblance. Do not
compare raw utility values across different gamma settings as if the objective
were unchanged.

Each unique run directory contains:

- `results.json`: full config, trial collections/partitions, selections, stage
  diagnostics, mean/sample-standard-deviation summaries, library versions,
  source-code hash, data hashes, and process-wide peak RSS where available;
- `metrics.csv`: per-trial logdets, gap, SLS nearest-neighbor distances, legacy
  Jaccard, cardinalities, proposal counts,
  timings, and estimated communication payloads;
- `config.yml`: the resolved configuration snapshot;
- `search_history.csv`: logdet traces when enabled (monotone for SLS only).

New SLS runs additionally contain a `report.md`; sweep summaries and primary
plots use Hamming distance. Exact/MCMC reporting is unchanged by this update.

Generated results and the virtual environment are excluded by `.gitignore`.
The central reference uses a dense global kernel. The distributed path builds
only client-local and server-pool-plus-context kernels; it does not use that global matrix
to optimize local/server selections. This remains a small-data reference,
not a scalable transport implementation.

Clients upload suggestion IDs and, in pooled mode, context IDs; the server
broadcasts its final set. Default uploads are 4,000 suggestion bytes plus 800
context bytes; the 100-item broadcast to 10 clients adds 8,000 bytes, for a
12,800-byte estimated total over two logical rounds. Payload byte counts are
estimates excluding network/protocol overhead. Clients
run sequentially in this local simulation. The ideal-parallel time estimate
assumes simultaneous clients and zero network overhead; it is not a measured
federated speedup. Peak RSS covers the entire script, not separate approach
peaks. The public random partitions are experiment data; membership of the
10-item $G_{0,i}$ inside each partition is disclosed in the current pooled mode.
All clients participate, candidate partitions are IID random and disjoint,
there are no failures or collusion defenses, and no noise, secure aggregation,
or compression beyond sending suggestion IDs. This is an independently
computed suggestion-aggregation baseline, not a solution to the repository's
full private federated problem. With one client, pooled context becomes that
client's context; without candidate compression, both final searches reduce
to the same single-context semi-supervised reference. With compression, their
eligible additions differ, so they need not return the same subset.

## Verification

Tests cover configuration errors, unique sampling, balanced partitions, kernel
PSD and singular fixtures, explicit jitter, tiny-set normalization and
enumerated objective bounds, monotone swaps, patience and tie semantics,
deterministic runs, public index mapping, server eligibility, Jaccard, output
preservation, and agreement between backend adapters. A one-client run with
no compression matches centralized selection under paired initialization.
Forced-set tests check immutable contexts, no overlap, joint scoring, empty
contexts, singular contexts, Schur-complement agreement via linear solves,
conditional tiny-set normalization, and monotone swaps in $G_1$ only.
Pooled-mode tests additionally protect shared-context equality, joint scoring,
explicit disclosure metadata, extra communication costs, and one-client
reduction without compression.
