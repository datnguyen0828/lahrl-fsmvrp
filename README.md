# FSMVRP Hierarchical PPO

This repository contains a PyTorch implementation of a hierarchical reinforcement learning solver for the Fleet Size and Mix Vehicle Routing Problem (FSMVRP). The solver uses a Semi-Markov Decision Process (SMDP) environment with two action levels:

- Fleet action: select which vehicle type should serve the next route.
- Route action: select which customer node should be visited next by the active vehicle.

The training code is based on PPO-style policy optimization, POMO rollout evaluation, a learned critic, future-cost estimation, entropy regularization, validation rollouts, and optional Optuna hyperparameter search.

## Problem Overview

FSMVRP extends the capacitated vehicle routing problem by allowing multiple vehicle types. Each vehicle type has:

- A capacity.
- A fixed usage cost.
- A variable distance cost.

The objective is to serve every customer exactly once while minimizing total routing cost:

```text
total_cost = sum(vehicle_fixed_cost + route_distance * vehicle_variable_cost)
```

The current environment assumes Euclidean distances computed from depot and customer coordinates.

## Repository Structure

```text
.
|-- FSMVRP_Env.py              # SMDP environment and rollout state definitions
|-- FSMVRP_Model.py            # Encoder, fleet decoder, and route decoder
|-- FSMVRP_Trainer.py          # PPO trainer, critic, validation, checkpointing
|-- FSMVRP_Tester.py           # Inference/test runner and CSV solution export
|-- FSMVRP_generate_data.py    # Utility for generating saved test tensors
|-- problemdef.py              # Random problem generation and 8-fold augmentation
|-- utils.py                   # Logging, metrics, plotting, and helper utilities
|-- test_20node.py             # Example 20-node inference script
|-- train/
|   |-- train_20node.py        # 20-node training entry point
|   `-- train_50node.py        # 50-node training entry point
|-- optuna/
|   |-- 20node_optuna_auto.py  # 20-node Optuna HPO entry point
|   `-- 50node_optuna_auto.py  # 50-node Optuna HPO entry point
|-- data/                      # Saved test tensors
|-- results/                   # Example checkpoints and training artifacts
|-- optuna_results/            # Existing Optuna SQLite study files
`-- log_image_style/           # Plot style JSON files for training curves
```

## Main Features

- Hierarchical policy with separate fleet and route decisions.
- Transformer-style node encoder with vehicle-aware decoding.
- PPO clipping for both fleet and route policies.
- Critic network for state-value learning.
- Future-cost estimator loss for long-horizon cost awareness.
- Entropy bonus and annealing support.
- POMO-style parallel rollouts.
- 8-fold geometric augmentation for evaluation.
- Validation during training with best-checkpoint saving.
- CSV export of inferred routes and per-route costs.
- Optuna-based hyperparameter optimization workflows.
- Single-GPU and distributed multi-GPU support for the 50-node training script.

## Requirements

### Hardware

- CPU-only execution is possible for small debugging runs, but full training is designed for NVIDIA GPUs.
- CUDA-capable GPU is strongly recommended.
- The 50-node configuration can require substantial VRAM. The script enables gradient checkpointing to reduce memory usage.

### Software

- Python 3.9 or newer is recommended.
- PyTorch with CUDA support if training on GPU.
- Windows, Linux, or macOS should work for the core Python code. Some command examples below use Windows PowerShell because this repository is currently laid out with Windows paths.

### Python Packages

Install the required packages:

```bash
pip install -r requirements.txt
```

Optional packages for Optuna AutoSampler workflows:

```bash
pip install optunahub cmaes scipy
```

If you need a specific CUDA-enabled PyTorch build, install PyTorch from the official selector first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone or copy this repository.
2. Create and activate a virtual environment.
3. Install the dependencies listed above.
4. Confirm that PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

5. Run scripts from the repository root unless you intentionally modify paths.

## Data Format

Saved `.pt` files contain a dictionary with the following tensors:

- `depot_xy`: depot coordinates with shape `(batch, 1, 2)`.
- `node_xy`: customer coordinates with shape `(batch, problem_size, 2)`.
- `node_demand`: customer demand with shape `(batch, problem_size)`.
- `agent_capacity`: vehicle capacity by vehicle type with shape `(batch, agent_num)`.
- `agent_fixed_cost`: fixed vehicle cost with shape `(batch, agent_num)`.
- `agent_variable_cost`: distance-based vehicle cost with shape `(batch, agent_num)`.

The included `data/` directory contains fixed test tensors for different problem sizes, for example:

```text
data/test_tensor(20)_6_100_1234.pt
data/test_tensor(50)_6_100_1234.pt
data/test_tensor(100)_6_100_1234.pt
```

## Generate Test Data

To generate a new saved test tensor, edit the parameters in `FSMVRP_generate_data.py` or call `generate_data(...)` directly:

```bash
python FSMVRP_generate_data.py
```

By default, this generates a 20-node test file with 100 instances and seed `1234`.

## Training

### Train the 20-node model

```bash
python train/train_20node.py
```

Default 20-node settings:

- Problem size: 20 customers.
- Vehicle types: 3 to 6.
- POMO size: 20.
- Epochs: 500.
- Training episodes per epoch: 10,000.
- Batch size: 128.
- Validation enabled with 8-fold augmentation.

Outputs are written under:

```text
results/train_20nodes/run_XXXX/
```

Each run stores:

- `config.json`: complete run configuration snapshot.
- `summary.json`: final best metric summary.
- `log.txt`: training log.
- `checkpoint-best.pt`: best validation checkpoint.
- `checkpoint-{epoch}.pt`: periodic checkpoints.
- `latest-tr_loss.jpg` and other plotted curves.

The latest 20-node run is tracked in:

```text
results/train_20nodes/latest_run.txt
```

### Train the 50-node model

```bash
python train/train_50node.py
```

Default 50-node settings:

- Problem size: 50 customers.
- Vehicle types: 3 to 6.
- POMO size: 50.
- Epochs: 2,000.
- Training episodes per epoch: 10,000.
- Batch size: 256.
- Gradient checkpointing enabled.
- Validation enabled.

Outputs are written under:

```text
results/train_50nodes/
```

### Multi-GPU training for 50 nodes

The 50-node script supports PyTorch distributed execution. Example:

```bash
torchrun --nproc_per_node=2 train/train_50node.py
```

The script reads `LOCAL_RANK` and `WORLD_SIZE`, initializes `nccl` when more than one process is used, and offsets the random seed per GPU.

## Inference and Testing

Run the 20-node tester:

```bash
python test_20node.py
```

The default tester configuration loads:

```text
results/train_20nodes/checkpoint-best.pt
data/test_tensor(20)_6_100_1234.pt
```

The tester reports:

- Raw score without augmentation.
- Best score with 8-fold augmentation.
- Optional per-solution route details.
- Optional CSV exports.

Default CSV output paths are configured in `test_20node.py`:

```text
result/test_20nodes/solution_summary_demo.csv
result/test_20nodes/solution_routes_demo.csv
```

## Checkpoints

Checkpoints are PyTorch `.pt` files containing:

- `epoch`: saved epoch.
- `model_state_dict`: policy model weights.
- `critic_state_dict`: critic weights.
- `optimizer_state_dict`: optimizer state.
- `scheduler_state_dict`: scheduler state.
- `result_log`: metric history.

To resume training, edit the `model_load` block in the relevant training script:

```python
'model_load': {
    'enable': True,
    'path': './results/train_20nodes',
    'epoch': 'best',
}
```

For 20-node runs that use the newer `run_XXXX` layout, `train_20node.py` can resolve the latest run when `model_load.path` points to the shared `results/train_20nodes` directory.

## Hyperparameter Optimization

Run Optuna HPO for 20 nodes:

```bash
python optuna/20node_optuna_auto.py
```

Run Optuna HPO for 50 nodes:

```bash
python optuna/50node_optuna_auto.py
```

The Optuna scripts define:

- Base environment, model, optimizer, and trainer parameters.
- Search spaces for learning rate, weight decay, PPO epsilon, PPO epochs, entropy weight, scheduler gamma, and related PPO parameters.
- SQLite storage under `results/optuna/`.
- Trial directories under `results/optuna/train_*_auto/run_XXXX/trials/`.

The AutoSampler path requires:

```bash
pip install optunahub cmaes scipy
```

## Configuration Guide

Most behavior is controlled directly in the entry-point scripts.

### Environment parameters

```python
env_params = {
    'min_problem_size': 20,
    'max_problem_size': 20,
    'pomo_size': 20,
    'min_agent_num': 3,
    'max_agent_num': 6,
}
```

Use these values to change problem size, POMO parallelism, or the range of available vehicle types.

### Model parameters

```python
model_params = {
    'embedding_dim': 128,
    'encoder_layer_num': 6,
    'head_num': 8,
    'qkv_dim': 16,
    'ff_hidden_dim': 512,
    'logit_clipping': 10.0,
    'future_beta': 0.0,
    'eval_type': 'softmax',
}
```

Use these values to change model capacity, attention dimensions, route selection behavior, and future-cost weighting.

### PPO parameters

```python
'ppo': {
    'epsilon': 0.2,
    'ppo_epochs': 6,
    'gamma': 0.99,
    'lambda_future': 0.5,
    'alpha_entropy': 0.05,
    'c_critic': 0.5,
}
```

These control PPO clipping, discounting, future-cost loss, entropy regularization, and critic loss weight.

### Validation parameters

```python
'validation': {
    'enable': True,
    'episodes': 400,
    'batch_size': 100,
    'seed': 9999,
    'aug_factor': 8,
    'objective_metric': 'val_softmax_aug_raw_score',
}
```

Validation determines which checkpoint is saved as `checkpoint-best.pt`.

## Output Interpretation

The objective is cost minimization. Lower scores are better.

Common metrics include:

- `train_score`: average training rollout cost.
- `train_loss`: PPO training loss.
- `val_softmax_aug_raw_score`: validation score using softmax sampling and augmentation.
- `val_argmax_aug_raw_score`: validation score using greedy action selection and augmentation.
- `best_val_score`: best validation objective observed during training.

## Common Workflow

```bash
# 1. Create and activate your Python environment.
python -m venv .venv

# 2. Install dependencies.
pip install -r requirements.txt

# 3. Generate or choose a compatible .pt dataset.
python FSMVRP_generate_data.py

# 4. Train a learned solver.
python train/train_20node.py

# 5. Evaluate the trained checkpoint.
python test_20node.py
```

## Notes and Caveats

- Most experiment settings are stored directly in Python dictionaries inside the entry-point scripts.
- Review `env_params`, `model_params`, `optimizer_params`, and `trainer_params` before long training runs.
- The model expects coordinate-based Euclidean instances with the six tensor fields described in the data format section.
- CUDA is enabled by default in the training scripts when `DEBUG_MODE = False`.
- The checked-in `results/` directory contains example checkpoints and plots. For a clean release, consider whether large generated artifacts should be kept in the repository or stored externally.

## Troubleshooting

### CUDA is not available

Check your PyTorch installation:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, install a CUDA-enabled PyTorch build or set the script to CPU/debug mode for small checks.

### Out-of-memory errors

Reduce one or more of:

- `train_batch_size`
- `pomo_size`
- `validation.batch_size`
- `validation.episodes`
- `embedding_dim`
- `encoder_layer_num`

For larger problem sizes, keep gradient checkpointing enabled.

### Checkpoint path errors

Verify that the configured folder contains the expected file:

```text
checkpoint-best.pt
checkpoint-10.pt
checkpoint-20.pt
```

Then update the `model_load.path` and `model_load.epoch` fields in the train or test script.

## License

This project is released under the MIT License. See `LICENSE` for details.

Some utility code includes MIT-license text from Yeong-Dae Kwon's POMO-style project utilities, preserved in `utils.py`.
