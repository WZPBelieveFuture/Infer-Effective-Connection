# Combined Kuramoto + Lorzen + real-fMRI Project

This folder merges the `combine_pro` codebase with the `paper_real_fmri` workflow into one compatible project.

## What Is Combined

- `kuramoto` workflow: generate Kuramoto data and train with the original coarse-graining pipeline.
- `lorzen` workflow: train on an existing dataset and optionally compute Lorenz-style causal graph metrics.
- `real_fmri` workflow: train on the paper fMRI dataset and run Jacobian-based ROI causal analysis.
- Shared entrypoints: `NIS_macro.py` and `NIS_micro.py`.

## Data Sources

`--data_source` now supports:

- `kuramoto`: generate/load Kuramoto data.
- `existing`: load an external `.csv`, `.npy`, or `.npz` dataset via `--data_path`.
- `lorzen`: alias of `existing`, kept for backward compatibility.
- `real_fmri`: alias of `existing`, with `auto` graph evaluation disabled by default.

Notes:

- Kuramoto examples use `./loc_data_kuramoto/generated_data.npz`.
- Lorzen examples in this merged project use `./loc_data_lorzen/generated_data.npz`.
- real-fMRI examples in this merged project use `./loc_data_real_fmri/generated_data.npz`.
- Macro group schedules are enabled automatically for Kuramoto and disabled automatically for Lorzen/existing data unless you override them.

## New Compatibility Flags

- `--min_dim`: lets one codebase cover both the Kuramoto and Lorzen latent-size defaults.
- `--causal_graph_mode {auto,none,lorenz96,macro_lorenz96}`:
  - `auto` enables Lorenz-style graph evaluation for stage-2 runs on `existing` / `lorzen` data.
  - `auto` becomes `none` for `real_fmri`, because that workflow has no Lorenz ground-truth graph.
  - `none` disables graph evaluation.
- `--group_schedule_mode {auto,on,off}` and `--group_schedule_path`: control whether macro grouping is loaded from `group.csv`.

## Recommended Scripts

Kuramoto:

- `run_kuramoto_stage1_macro.sh`
- `run_kuramoto_stage2_macro.sh`
- `run_kuramoto_stage2_micro.sh`

Lorzen:

- `run_lorzen_stage2_macro.sh`
- `run_lorzen_stage2_micro.sh`

real-fMRI:

- `run_real_fmri_stage1_macro.sh`
- `run_real_fmri_stage2_macro.sh`
- `run_real_fmri_jacobian.sh`
- `run_real_fmri_ppt.sh`

Legacy Kuramoto scripts are also kept:

- `run_mice_stage1_macro.sh`
- `run_mice_stage2_macro.sh`
- `run_mice_stage2_micro.sh`

## Example Commands

Kuramoto stage 1 macro:

```sh
python NIS_macro.py --data_source kuramoto --train_stage 1 --run_name stage1_macro
```

Lorzen stage 2 macro:

```sh
python NIS_macro.py \
  --data_source lorzen \
  --data_path ./loc_data_lorzen/generated_data.npz \
  --train_stage 2 \
  --scale_id 0 \
  --min_dim 8 \
  --causal_graph_mode auto \
  --group_schedule_mode off \
  --run_name stage2_lorzen_macro
```

Lorzen stage 2 micro:

```sh
python NIS_micro.py \
  --data_source lorzen \
  --data_path ./loc_data_lorzen/generated_data.npz \
  --train_stage 2 \
  --scale_id 0 \
  --causal_graph_mode auto \
  --run_name stage2_lorzen_micro
```

real-fMRI stage 2 macro:

```sh
python NIS_macro.py \
  --data_source real_fmri \
  --data_path ./loc_data_real_fmri/generated_data.npz \
  --train_stage 2 \
  --scale_id 0 \
  --group_schedule_mode off \
  --causal_graph_mode none \
  --min_dim 8 \
  --run_name stage2_real_fmri_macro
```

real-fMRI Jacobian analysis:

```sh
python compute_jacobian_matrix.py \
  --run_name stage2_real_fmri_macro \
  --data_path ./loc_data_real_fmri/generated_data.npz \
  --sample_count 100

python generate_causal_comparison_ppt.py \
  --mode real_fmri_roi \
  --run_name stage2_real_fmri_macro \
  --sample_count_tag 100 \
  --scale_id 0
```
