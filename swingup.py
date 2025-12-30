# swingup.py
import gymnasium as gym
import numpy as np
import math

class CartPoleSwingUp(gym.Wrapper):
    """
    CartPole dynamics + swing-up reward + downward reset.

    - theta wrapped to [-pi, pi]
    - reset near DOWN: theta = pi + U(-theta_width, theta_width)
    - reward = (cos(theta)+1)/2 + upright-stability bonus
    - termination: cart out of bounds (terminated=True)
    - time limit: truncated=True when steps>=max_steps (like Gym TimeLimit)
    """
    def __init__(self, env, max_steps=500):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

        # allow full rotation
        self.env.unwrapped.theta_threshold_radians = math.pi

        # match notebook's x_threshold
        self.env.unwrapped.x_threshold = 2.4
        self.x_threshold = self.env.unwrapped.x_threshold

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            math.pi,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        # width is in radians, default ~0.1 rad
        theta_width = 0.1
        if options is not None and "theta_width" in options:
            theta_width = float(options["theta_width"])

        rng = self.env.unwrapped.np_random

        # ---- START NEAR DOWN (pi) ----
        theta = math.pi + rng.uniform(-theta_width, +theta_width)
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi

        # small randomization like the notebook
        x = rng.uniform(-0.1, 0.1)
        x_dot = rng.uniform(-0.1, 0.1)
        theta_dot = rng.uniform(-0.1, 0.1)

        self.env.unwrapped.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps = 0

        obs = np.array(self.env.unwrapped.state, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.steps += 1

        x, x_dot, theta, theta_dot = obs

        # wrap theta
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
        self.env.unwrapped.state[2] = theta
        obs = np.array(self.env.unwrapped.state, dtype=np.float32)

        # ---- reward EXACTLY like notebook ----
        reward = (np.cos(theta) + 1.0) / 2.0
        if abs(theta) < 0.1 and abs(theta_dot) < 0.5:
            reward += 1.0

        # ---- termination/truncation like notebook ----
        out_of_bound = abs(x) > self.x_threshold
        time_limit = self.steps >= self.max_steps

        terminated = bool(out_of_bound)      # failure
        truncated = bool(time_limit and not terminated)  # time limit

        return obs, float(reward), terminated, truncated, info


def make_swingup_cartpole(max_steps=500, **make_kwargs):
    base_env = gym.make("CartPole-v1", **make_kwargs)
    return CartPoleSwingUp(base_env, max_steps=max_steps)
