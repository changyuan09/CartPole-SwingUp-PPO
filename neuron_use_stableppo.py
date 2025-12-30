# run_sb3_policy.py
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# IMPORTANT: your env id must be registered the same way as in training
# If you used a custom env class, you must register it again here.
# If you used plain "CartPole-v1" wrapped, recreate the same wrapper here.

from swingup import make_swingup_cartpole  # your wrapper factory

MODEL_ZIP = "ppo_grpo_cartpole_swingup_checkpoint.zip"
RENDER = True

def main():
    model = PPO.load(MODEL_ZIP, device="cpu")  # or "cuda" if you have it

    env = make_swingup_cartpole(max_steps=1000, render_mode="human" if RENDER else None)

    obs, info = env.reset(seed=np.random.randint(0, 2**31 - 1))
    done = False
    ep_return = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # action is int for discrete
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_return += reward
        steps += 1

        if RENDER:
            time.sleep(0.02)

        done = terminated or truncated

    print(f"Episode finished: return={ep_return:.3f}, steps={steps}")
    env.close()

if __name__ == "__main__":
    while True:
        main()
