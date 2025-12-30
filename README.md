# CartPole SwingUp with PPO (Neural-Network-from-Scratch)

This repo trains a **PPO (Proximal Policy Optimization)** policy to solve **CartPole SwingUp**: starting near the *downward* pose (around ±π), the agent learns to swing the pole up and balance it by pushing the cart left/right.

This project uses a small custom MLP-style “neuron network” (policy + value) and runs **8 parallel environments** using **multi-core processing (8 cores)** to speed up rollout collection.

---

## Project layout

Typical structure:

├── main.py
├── neuron_set.py
├── neuron_use.py
├── neuron_use_stableppo.py
├── requirements.txt
├── weights/
│ ├── policy_.
│ └── value_.
└── (other helper files, logs, etc.)


### What each script does

- **`main.py`**  
  The main PPO training script. It spawns **8 environments in parallel** using multi-core processing (intended for an 8-core CPU setup).  
  Each environment reset initializes the pole angle with randomness: the pole starts near the *downward* configuration with an offset of **±5 degrees** around **+π and -π** (i.e., near upside-down / hanging pose).

- **`neuron_set.py`**  
  Initializes the neuron networks **randomly** before training. Run this first if you want to train from scratch.

- **`weights/`**  
  After training, the trained parameters are saved here:
  - **policy network** weights (used to pick left/right actions)
  - **value network** weights (used to estimate V(s) during PPO)

- **`neuron_use.py`**  
  Loads the trained weights from `weights/` and runs inference to demonstrate the learned swing-up policy.

- **`neuron_use_stableppo.py`**  
  Uses a **pre-trained** (already trained) version so you **don’t need to train yourself**.  
  Note: the policy is not perfect—**some runs may fail** to swing up. Re-run to see successful swing-up behavior.

---

## Network architecture (important details)

### Policy network
- **Inputs:** 4 observation values  
- **Hidden layers:** 2 layers, **5 hidden neurons each**
- **Outputs:** 2 outputs → interpreted as **Left** vs **Right** action

### Value network
- **Inputs:** 4 observation values  
- **Hidden layers:** 2 layers, **5 hidden neurons each**
- **Output:** 1 neuron → predicts **value estimate** V(s) given observations

---

## Requirements / installation

### 1) (Recommended) Create a virtual environment
From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
2) Install dependencies
pip install -r requirements.txt


Do not commit .venv/ to GitHub. Virtual environments contain large binaries and will cause push failures.

Run the pre-trained demo (no training required)

If you just want to see swing-up behavior immediately:

python3 neuron_use_stableppo.py


This uses a pre-trained PPO agent. It is not guaranteed to succeed every run, so if it fails, just run it again.

Train from scratch (full workflow)
Step 1 — Initialize random neuron networks
python3 neuron_set.py


This creates/initializes both the policy and value networks with random weights so PPO training can start from scratch.

Step 2 — Train PPO with 8 parallel environments
python3 main.py


main.py runs PPO training and spawns 8 parallel env processes (multi-core).
During resets, the pole angle is randomized within ±5° around +π and -π (downward pose).
After training completes, learned weights are saved under:

weights/

Step 3 — Run inference with your trained weights
python3 neuron_use.py


This loads the trained policy network (and any required saved artifacts) from weights/ and runs the environment to demonstrate the learned swing-up policy.

Notes / common issues

GitHub push failed because of .venv/
You must NOT commit .venv/. Add it to .gitignore:

.venv/


If you already committed it, remove it from Git history before pushing.

Swing-up sometimes fails
This is expected. Swing-up is sensitive to initialization, randomness, and policy quality. Re-run inference a few times.

Parallel environments
main.py is designed around 8 parallel envs. If your CPU has fewer cores, it may run slower.

Quick command summary
Install
pip install -r requirements.txt

Pre-trained demo (no training)
python3 neuron_use_stableppo.py

Train from scratch
python3 neuron_set.py
python3 main.py
python3 neuron_use.py