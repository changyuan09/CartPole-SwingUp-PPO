# CartPole SwingUp with PPO (Neural-Network-from-Scratch)

This repository trains a **PPO (Proximal Policy Optimization)** agent to solve **CartPole SwingUp**. The pole starts near the **downward** configuration (around **±π**) and the agent learns to swing it up and balance it by pushing the cart **left/right**.

Key features:
- Custom lightweight MLP-style “neuron network” implementation (policy + value)
- **8 parallel environments** via **multi-core processing (8 cores)** for faster rollouts

---

## Project layout

```text
.
├── main.py
├── neuron_set.py
├── neuron_use.py
├── neuron_use_stableppo.py
├── requirements.txt
├── weights/
│   ├── policy_*
│   └── value_*
└── (other helper files, logs, etc.)
```

---

## Scripts

### `main.py`
Main PPO training script.

- Spawns **8 environments in parallel** (intended for an 8-core CPU setup).
- On reset, initializes the pole angle near the **downward** pose with randomness:
  - randomized within **±5°** around **+π** and **-π**.

### `neuron_set.py`
Initializes the policy/value neuron networks with **random weights**.

Run this first if you want to train from scratch.

### `weights/`
Training outputs are saved here:
- policy network weights (action selection)
- value network weights (V(s) estimation used by PPO)

### `neuron_use.py`
Loads trained weights from `weights/` and runs inference to demonstrate swing-up behavior.

### `neuron_use_stableppo.py`
Runs a **pre-trained** PPO agent so you can demo without training.

Note: swing-up is not guaranteed every run—re-run if it fails.

---

## Network architecture

### Policy network
- Inputs: **4** observation values
- Hidden layers: **2** layers, **5** hidden neurons each
- Outputs: **2** outputs → interpreted as **Left** vs **Right**

### Value network
- Inputs: **4** observation values
- Hidden layers: **2** layers, **5** hidden neurons each
- Output: **1** neuron → predicts **V(s)**

---

## Installation

### 1) Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

**Do not commit** `.venv/` to GitHub. Add it to `.gitignore` (see below).

---

## Usage

### Run the pre-trained demo (no training required)

```bash
python3 neuron_use_stableppo.py
```

If swing-up fails, run it again.

### Train from scratch (full workflow)

#### Step 1 — Initialize random networks

```bash
python3 neuron_set.py
```

#### Step 2 — Train PPO (8 parallel envs)

```bash
python3 main.py
```

Notes:
- Uses **8 parallel environment processes**
- Reset randomizes pole angle within **±5°** around **+π** and **-π**
- Saves learned weights under `weights/`

#### Step 3 — Run inference using your trained weights

```bash
python3 neuron_use.py
```

---

## Notes / common issues

### GitHub push failed because of `.venv/`
Don’t commit the virtual environment. Add this to `.gitignore`:

```gitignore
.venv/
```

If you already committed it, remove it from Git history before pushing.

### Swing-up sometimes fails
Expected: swing-up can be sensitive to initialization, randomness, and policy quality. Re-run inference a few times.

### Parallel environments
`main.py` assumes **8 parallel envs**. Fewer CPU cores will generally run slower.

---

## Quick command summary

### Install
```bash
pip install -r requirements.txt
```

### Pre-trained demo (no training)
```bash
python3 neuron_use_stableppo.py
```

### Train from scratch
```bash
python3 neuron_set.py
python3 main.py
python3 neuron_use.py
```
