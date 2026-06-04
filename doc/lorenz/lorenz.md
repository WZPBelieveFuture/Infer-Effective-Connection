# Lorenz-96 合成数据

本文档说明 `generate_lorenz_data.ipynb` 中的 Lorenz-96 数据生成器。

## 来源示例

Lorenz-96 模型由 Edward N. Lorenz 提出，常作为数值天气预报、混沌动力系统、数据同化和因果发现实验中的标准合成系统。该模型用一个带周期边界的一维环形变量序列表示空间站点，每个站点同时受到平流项、耗散项和外部强迫项影响。

本 notebook 使用的是单尺度 Lorenz-96 模型：

- `m = 8` 个组级变量，也就是 8 个 Lorenz-96 站点。
- 外部强迫 `F = 10.0`。
- 连续时间区间为 `t_span = (0, 50)`。
- 先在 `T * 10` 个时间点上高分辨率积分，再每 10 个点降采样一次。
- 默认最终时间序列长度 `T = 1000`。
- 积分器为 `scipy.integrate.solve_ivp`，方法 `RK45`，容差 `rtol = 1e-8`、`atol = 1e-10`。

参考资料：

- Lorenz 原始报告：<https://www.ecmwf.int/en/elibrary/75462-predictability-problem-partly-solved>
- Lorenz-96 模型综述：<https://arxiv.org/abs/2005.07767>

## Lorenz-96 模型

令 $Y_i(t)$ 表示第 $i$ 个组级变量在连续时间 $t$ 的状态，$i = 0, \dots, m-1$。notebook 中使用的 Lorenz-96 方程为：

$$
\frac{dY_i}{dt}
= \left(Y_{i+1} - Y_{i-2}\right)Y_{i-1} - Y_i + F
$$

其中所有下标都按模 $m$ 取值，因此变量排成一个环。例如 $Y_{-1}$ 表示 $Y_{m-1}$，$Y_m$ 表示 $Y_0$。

方程中的各项含义为：

- $\left(Y_{i+1} - Y_{i-2}\right)Y_{i-1}$ 是非线性平流项。
- $-Y_i$ 是自耗散项。
- $F$ 是常数外部强迫项。notebook 默认 `F = 10.0`，会产生混沌动力学。

notebook 的默认生成过程是：

1. 用 `np.random.randn(m) * 0.1` 生成初始状态 `y0`。
2. 调用 `solve_ivp` 在 `(0, 50)` 上积分 Lorenz-96 常微分方程。
3. 从高分辨率解 `sol.y` 中取 `sol.y[:, ::10]` 得到长度为 `T` 的组级序列。
4. 对每个组级时间序列单独做零均值、单位标准差标准化：

$$
Y_i \leftarrow \frac{Y_i - \operatorname{mean}(Y_i)}
{\operatorname{std}(Y_i) + 10^{-8}}
$$

## 因果矩阵

Lorenz-96 的真实组级因果关系由方程右侧的直接依赖决定。第 $i$ 个目标变量的导数依赖于：

- 自身 $Y_i$；
- 前一个变量 $Y_{i-1}$；
- 前两个变量 $Y_{i-2}$；
- 后一个变量 $Y_{i+1}$。

因此真实因果矩阵 `G` 定义为：

$$
G_{i,j} =
\begin{cases}
1, & j \in \{i, i-1, i-2, i+1\}\pmod m, \\
0, & \text{otherwise}.
\end{cases}
$$

矩阵的行表示目标组，列表示源组。对角线为 1，因为每个变量都有自耗散项 `-Y_i`。

在默认 `m = 8` 的设置下：

- 每个目标组有 1 条自因果边和 3 条邻域因果边。
- 包含自边的总边数为 `8 * 4 = 32`。
- 非对角边数为 `8 * 3 = 24`。

项目中的 `src/models_macro.py` 和 `src/models_micro.py` 使用同样的规则构造 Lorenz-96 真实图：`build_lorenz96_adjacency` 会把 `i`、`i-1`、`i-2`、`i+1` 四个源位置标为 1。

## 分组观测层

Lorenz-96 模型本身生成的是组级变量 `Y_true`，形状为 `(m, T)`。为了配合项目中的分组观测变量设定，notebook 为每个组级变量生成多个低层观测变量。

每个组 `i` 的观测变量数为 `p_i`，由 `p_list` 记录。默认总观测变量数为：

```text
m * avg_vars_per_group = 8 * 4 = 32
```

notebook 中 `p_list` 的生成规则是：

- 每个组至少有 2 个观测变量。
- 前 `m - 1` 个组的 `p_i` 从 `[2, 2 * avg_vars_per_group)` 随机抽样，并调整为偶数。
- 最后一个组吸收剩余变量数，使总观测维度保持为 32。

需要注意，`p_list` 的抽样发生在 `np.random.seed(42)` 之前，因此如果重新运行 notebook，具体的 `p_list` 可能会随当前 NumPy 随机状态变化。实际训练时应以保存文件中的 `group` 字段为准。

对每个组，观测层为：

$$
X_i = B_i Y_i + \epsilon_i
$$

其中：

- `X_i` 是第 `i` 个组的观测矩阵，形状为 `(p_i, T)`。
- `B_i` 是随机混合向量，生成方式为 `np.random.randn(p_i, 1) * 0.5 + 1.0`，随后按二范数归一化。
- $\epsilon_i$ 是独立高斯噪声，notebook 中每个观测变量加入 `np.random.randn(T) * 0.05`。
- 每个观测变量生成后会再次单独标准化。

最终所有组的观测变量按组顺序拼接：

```python
X_new = np.concatenate(X, axis=0).T
```

因此默认 `X_new` 的形状为 `(1000, 32)`：行是时间点，列是观测变量。列顺序与 `group` 中的组大小连续对应。

## 保存数据格式

notebook 写入：

```text
loc_data_lorenz/generated_data.npz
```

下游训练所需字段：

- `data`: 观测数据，形状为 `(T, sum(group))`，默认是 `(1000, 32)`。
- `group`: 每个组包含的观测变量数，形状为 `(m,)`。

notebook 中 `Y_true` 是组级 Lorenz-96 真实状态，但当前实际保存到 `loc_data_lorenz/generated_data.npz` 的字段只包含 `data` 和 `group`。如果需要同时保存真实组级状态、因果矩阵或其他诊断信息，需要在保存单元中额外加入这些字段。

下游读取方式：

- `src/data_sources.py` 从 `.npz` 中读取 `data` 字段。
- `NIS_macro.py` 和 `NIS_micro.py` 可以从 `.npz` 中读取 `group` 字段。
- Lorenz-96 的真实因果图不依赖 `.npz` 文件保存，而是由 `--causal_graph_mode lorenz96` 或 `--causal_graph_mode macro_lorenz96` 在模型代码中按上述邻域规则生成。

notebook 末尾还会把同一份 `X_new` 和 `group` 写入 `syn_data/*.csv`，包括 `all_data.csv`、`train_input.csv`、`train_target.csv`、`test_input.csv`、`test_target.csv` 和 `group.csv`。
