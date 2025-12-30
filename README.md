# CartPole SwingUp with PPO

This project trains a Proximal Policy Optimization (PPO) agent to solve the **CartPole SwingUp** task: starting from a hanging pole, the agent learns to swing the pole upward and stabilize it by controlling the cart.

## What’s inside
- PPO training loop (policy + value network)
- Environment interaction + rollout collection
- Checkpoint saving (optional) and evaluation/render (if provided)

## Quick start
1) Install dependencies:
```bash
pip install -r requirements.txt
