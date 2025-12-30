# run_cartpole_policy_single.py

import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from swingup import make_swingup_cartpole

from neuron_set_1 import PolicyNet  # your network definition

WEIGHTS_PATH = "Weights/Policy_Weights.pth"
RENDER = True   # set False if running headless or no GUI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_policy(weights_path):
    # 🔹 Always use CPU on MacBook
    print("Using device:", device)

    policy = PolicyNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    # 🔽 print all params
    print("\n=== Loaded policy parameters ===")
    for name, param in policy.named_parameters():
        print(f"{name}:\n{param.data}\n")
    print("=== End of parameters ===\n")
    # 🔼

    return policy


def select_action(policy, obs):
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs = policy(obs_tensor)
        action = torch.argmax(probs, -1).item()
    return action


def main():
    policy = load_policy(WEIGHTS_PATH)

    if RENDER:
        env = make_swingup_cartpole(max_steps=1000, render_mode="human")
    else:
        env = make_swingup_cartpole(max_steps=1000)


    # single episode
    seed = np.random.randint(0, 2**31 - 1)
    obs, info = env.reset(seed=seed)

    done = False
    ep_return = 0.0
    steps = 0

    while not done:
        action = select_action(policy, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        steps += 1

        if RENDER:
            time.sleep(0.02)

        if terminated or truncated:
            done = True

    print(f"Single episode finished: return = {ep_return}, steps = {steps}")
    env.close()


if __name__ == "__main__":
    while True: 
        main()


