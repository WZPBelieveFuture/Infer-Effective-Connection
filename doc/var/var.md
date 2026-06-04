# VAR Synthetic Data

This note documents the VAR data generator in `generate_var_data.ipynb`.

## Source Example

The group-level VAR example is adapted from Tank et al., *Neural Granger Causality*, which evaluates Granger-causal discovery on sparse VAR(1) and VAR(2) systems. The paper uses:

- `p = 20` time series.
- VAR lag order `K = 1` or `K = 2`.
- Time-series lengths `T = 250, 500, 1000`.
- Each target series depends on its own past values and on three randomly selected other source series.
- Each nonzero VAR coefficient is `0.1`.
- Results in the paper are averaged over five random initializations.

References:

- Paper: <https://arxiv.org/abs/1802.05842>
- Paper full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9739174/>
- Official implementation repository: <https://github.com/iancovert/Neural-GC>

## VAR Model

Let $Y_t$ be the group-level state vector at time $t$, with $m = 20$ groups. A VAR($K$) model is

$$
Y_t = \sum_{k=1}^{K} A^{(k)} Y_{t-k} + e_t
$$

where:

- $K$ is the lag order. In the paper's VAR experiments, $K = 1$ or $K = 2$.
- $k$ indexes the lag matrix, not a variable within a group.
- $A^{(k)}$ is the coefficient matrix for lag $k$.
- $A^{(k)}_{i,j} \neq 0$ means source group $j$ helps predict target group $i$ using lag $k$.
- $e_t$ is additive noise.

The notebook defaults to the VAR(2), `T = 1000` case:

$$
Y_t = A^{(1)}Y_{t-1} + A^{(2)}Y_{t-2} + e_t
$$

Set `CONFIG["var_lag"] = 1` to reproduce the VAR(1) case.

## Granger Causal Matrix

The ground-truth causal matrix `G` is defined by the nonzero VAR coefficients:

$$
G_{i,j} =
\begin{cases}
1, & \exists k \in \{1, \dots, K\}\ \text{such that}\ A^{(k)}_{i,j} \neq 0, \\
0, & \text{otherwise}.
\end{cases}
$$

Rows are target groups and columns are source groups. The diagonal is one because each target depends on its own past values.

With the default settings:

- `m = 20`
- Each target has one self cause and three extra causes.
- Total edges including self edges: `20 * 4 = 80`.
- Off-diagonal edges: `20 * 3 = 60`.

## Grouped Observation Layer

The VAR paper example is defined at the time-series level. This project expects grouped observed variables, so the notebook treats each VAR series as one group-level latent variable `Y_i` and creates multiple observed variables per group:

$$
X_i = B_i Y_i + \epsilon_i
$$

where:

- `X_i` is the observed matrix for group `i`, with shape `(p_i, T)`.
- `B_i` is a random `p_i x 1` mixing vector.
- `p_i` is sampled reproducibly so the average number of observed variables per group is 4.

This observation layer follows the same project convention used by the Lorenz generator: it gives the training code grouped low-level variables while keeping a known group-level causal graph.

## Saved Data Format

The notebook writes:

```text
loc_data_var/generated_data.npz
```

Required downstream fields:

- `data`: observed data with shape `(T, sum(group))`.
- `group`: group sizes, shape `(m,)`.

Ground-truth and diagnostic fields:

- `Y_true`: standardized group-level VAR states, shape `(m, T)`.
- `Y_var_raw`: raw VAR states before standardization, shape `(m, T)`.
- `coef_matrices`: VAR coefficient matrices, shape `(K, m, m)`.
- `causal_matrix`: binary Granger causal matrix, shape `(m, m)`.
- `measurement_matrix`: observed-variable to group-level mixing matrix, shape `(sum(group), m)`.
- `group_ids`: group id for each observed variable, shape `(sum(group),)`.
- `config_json`: JSON string recording the generation configuration.

The downstream loader in `src/data_sources.py` reads the `data` field from `.npz` files, and `NIS_macro.py` / `NIS_micro.py` can read the `group` field for group schedules.
