import time
import gymnasium as gym
import numpy as np
from neuron_set_1 import PolicyNet, ValueNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from swingup import make_swingup_cartpole
from gymnasium.vector import AsyncVectorEnv



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

current_policy = PolicyNet().to(device)
current_policy.load_state_dict(torch.load('Weights/Policy_Weights.pth', map_location=device))

old_policy = PolicyNet().to(device)
old_policy.load_state_dict(current_policy.state_dict())

value_net      = ValueNet().to(device)
value_net.load_state_dict(torch.load('Weights/Value_Weights.pth', map_location=device))


optimizer_value  = optim.Adam(value_net.parameters(), lr=1e-4)
optimizer_policy = optim.Adam(current_policy.parameters(), lr=1e-4)

rng = np.random.default_rng()


def main(theta_width_rad, *, n_envs: int = 8, n_steps: int = 2048):
    """
    Multi-core rollout using AsyncVectorEnv (one subprocess per env).

    Key behavior:
      - Each env gets its OWN random theta_width on every reset
        (sampled from Uniform(0, theta_width_rad) by default).
      - Collect exactly n_steps steps per env (total transitions = n_envs * n_steps).
      - Reset env immediately on terminated/truncated.
      - Flush partial trajectories at the end with bootstrap last_value.
    """
    global old_policy

    old_policy.load_state_dict(current_policy.state_dict())
    policy = old_policy
    policy.eval()
    value_net.eval()

    # ---- MULTICORE: each env runs in its own subprocess ----
    def make_thunk():
        return make_swingup_cartpole()

    vec_env = AsyncVectorEnv([make_thunk for _ in range(n_envs)])

    # per-env active traj dict
    traj_list = [{
        "obs": [],
        "actions": [],
        "rewards": [],
        "logp_old": [],
        "values": [],
        "terminated": [],
        "truncated": [],
        "last_value": 0.0,
    } for _ in range(n_envs)]

    trajectories = []

    # Helper: reset ONLY env e, with its own seed and its own theta_width
    def reset_one(e: int):
        seed_e = int(rng.integers(0, 2**31 - 1))

        # choose a per-env random width; change this distribution if you want
        width_e = float(rng.uniform(0.0, theta_width_rad))

        reset_mask = np.zeros(n_envs, dtype=np.bool_)
        reset_mask[e] = True

        seed_list = [None] * n_envs
        seed_list[e] = seed_e

        obs_reset, info_reset = vec_env.reset(
            seed=seed_list,
            options={"theta_width": width_e, "reset_mask": reset_mask},
        )
        return obs_reset[e]  # the new obs for env e

    # ---- INITIAL RESET ----
    # Need at least one reset to initialize the vector env.
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_envs)]
    obs, info = vec_env.reset(seed=seeds)  # options omitted here; we override per env below

    # Now apply per-env random width by resetting each env individually once.
    for e in range(n_envs):
        obs[e] = reset_one(e)

    # ---- FIXED steps per env ----
    for _t in range(n_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)  # [n_envs,4]
        with torch.no_grad():
            action_probs = policy(obs_tensor)               # [n_envs,2]
            v_t = value_net(obs_tensor).squeeze(-1)         # [n_envs]

        actions = torch.multinomial(action_probs, 1).squeeze(1)  # [n_envs]
        probs_a = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        logp_old = torch.log(probs_a + 1e-8)

        actions_np = actions.detach().cpu().numpy()
        logp_np    = logp_old.detach().cpu().numpy()
        v_np       = v_t.detach().cpu().numpy()

        # parallel step
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions_np)

        for e in range(n_envs):
            traj = traj_list[e]

            # store pre-step obs + action info
            traj["obs"].append(obs[e])
            traj["actions"].append(int(actions_np[e]))
            traj["logp_old"].append(float(logp_np[e]))
            traj["values"].append(float(v_np[e]))

            # store results
            traj["rewards"].append(float(rewards[e]))
            traj["terminated"].append(bool(terminated[e]))
            traj["truncated"].append(bool(truncated[e]))

            if terminated[e] or truncated[e]:
                # bootstrap only if truncated (time limit)
                if truncated[e] and (not terminated[e]):
                    last_obs_tensor = torch.tensor(next_obs[e], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        traj["last_value"] = float(value_net(last_obs_tensor).squeeze(-1).item())
                else:
                    traj["last_value"] = 0.0

                trajectories.append(traj)

                # reset this env with NEW random width + seed
                next_obs[e] = reset_one(e)

                # fresh traj dict
                traj_list[e] = {
                    "obs": [],
                    "actions": [],
                    "rewards": [],
                    "logp_old": [],
                    "values": [],
                    "terminated": [],
                    "truncated": [],
                    "last_value": 0.0,
                }

        obs = next_obs

    # ---- FLUSH partial trajectories at rollout end (bootstrap) ----
    for e in range(n_envs):
        traj = traj_list[e]
        if len(traj["rewards"]) == 0:
            continue

        last_obs_tensor = torch.tensor(obs[e], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            traj["last_value"] = float(value_net(last_obs_tensor).squeeze(-1).item())

        trajectories.append(traj)

    vec_env.close()
    return trajectories




def process(trajectories, gamma=0.99, lam=0.95):
    dataset = []
    traj_total_rewards = []

    for traj in trajectories:
        rewards = np.array(traj["rewards"], dtype=np.float32)
        values  = np.array(traj["values"], dtype=np.float32)
        terms   = np.array(traj["terminated"], dtype=np.bool_)
        truncs  = np.array(traj["truncated"], dtype=np.bool_)
        T = len(rewards)

        # ✅ log only true finished episodes (not rollout-end flush segments)
        is_episode = (T > 0) and (terms[-1] or truncs[-1])
        if is_episode:
            traj_total_rewards.append(float(rewards.sum()))

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            nonterminal = 0.0 if terms[t] else 1.0
            if t == T - 1:
                next_value = traj["last_value"]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values

        for t in range(T):
            dataset.append({
                "obs": traj["obs"][t],
                "action": traj["actions"][t],
                "logp_old": traj["logp_old"][t],
                "advantage": float(advantages[t]),
                "return": float(returns[t]),
            })

    # if no completed episodes happened in this rollout, keep avg/std sane
    if len(traj_total_rewards) == 0:
        avg_traj_return = 0.0
        std_traj_return = 0.0
    else:
        avg_traj_return = float(np.mean(traj_total_rewards))
        std_traj_return = float(np.std(traj_total_rewards))

    return dataset, avg_traj_return, std_traj_return



def train(dataset, *, batch_size: int = 64):
    """
    Minimal changes to match SB3 update schedule:
      - batch_size = 64  (default SB3 PPO)
      - num_epochs = 10  (default SB3 PPO)
    Your PPO loss is still your PPO loss (same structure).
    """
    global current_policy, value_net, optimizer_policy, optimizer_value

    eps = 1e-8

    obs_list = []
    action_list = []
    return_list = []
    adv_list = []
    logp_old_list = []

    for sample in dataset:
        obs_list.append(torch.tensor(sample["obs"], dtype=torch.float32, device=device))
        action_list.append(sample["action"])
        return_list.append(sample["return"])
        adv_list.append(sample["advantage"])
        logp_old_list.append(sample["logp_old"])

    obs_tensor      = torch.stack(obs_list)  # [N,4]
    action_tensor   = torch.tensor(action_list, dtype=torch.long, device=device)        # [N]
    return_tensor   = torch.tensor(return_list, dtype=torch.float32, device=device)     # [N]
    adv_tensor      = torch.tensor(adv_list, dtype=torch.float32, device=device)        # [N]
    logp_old_tensor = torch.tensor(logp_old_list, dtype=torch.float32, device=device)  # [N]

    # normalize advantages across the WHOLE batch
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + eps)

    value_net.train()
    current_policy.train()

    eps_clip = 0.2
    num_epochs = 10              # <-- SB3 default
    entropy_coef = 0.1
    value_coef   = 0.5

    N = obs_tensor.shape[0]
    if batch_size <= 0:
        batch_size = N

    last_loss = 0.0

    for epoch in range(num_epochs):
        perm = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]

            mb_obs      = obs_tensor[idx]
            mb_actions  = action_tensor[idx]
            mb_returns  = return_tensor[idx]
            mb_advs     = adv_tensor[idx]
            mb_logp_old = logp_old_tensor[idx]

            # ---- policy loss (PPO clip) ----
            action_probs = current_policy(mb_obs)  # [B,2]
            dist_probs   = action_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)  # [B]
            logp_new     = torch.log(dist_probs + 1e-8)

            ratio = torch.exp(logp_new - mb_logp_old)

            surr1 = ratio * mb_advs
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * mb_advs
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            policy_loss = policy_loss - entropy_coef * entropy

            # ---- value loss ----
            values_pred = value_net(mb_obs).squeeze(-1)
            value_loss  = F.mse_loss(values_pred, mb_returns)

            # ---- optimize policy ----
            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()

            # ---- optimize value ----
            optimizer_value.zero_grad()
            value_loss.backward()
            optimizer_value.step()

            last_loss = float(policy_loss.item() + value_coef * value_loss.item())

    return last_loss



if __name__ == "__main__":
    log_path = "train_log.csv"

    solve_threshold  = 800.0       # this is THE threshold now
    required_streak  = 30           # need 10 consecutive >= 1200
    streak           = 0            # current consecutive count

    # curriculum params
    theta_width_deg   = 5.0         # start: small band around upright (~5°)
    curriculum_stage  = 0           # 0 = tiny, 1 => 30°, 2 => 60°, ...

    episode = 0
    with open(log_path, "w") as f:
        f.write("episode,avg_return,loss,theta_width_deg\n")
        while True:
            theta_width_rad = math.radians(theta_width_deg)
            trajectories = main(theta_width_rad)
            dataset, avg, std = process(trajectories)
            loss = train(dataset)

            f.write(f"{episode},{avg:.6f},{loss:.6f},{theta_width_deg:.2f}\n")
            f.flush()

            print(
                f"[episode {episode}] "
                f"avg_return={avg:.3f}, loss={loss:.3f}, "
                f"theta_width={theta_width_deg:.1f}°"
            )

            # ---- curriculum / threshold logic (1200, 10 times) ----
            if avg >= solve_threshold:
                streak += 1
                print(f"  streak {streak}/{required_streak} episodes ≥ {solve_threshold}")
            else:
                streak = 0

            if streak >= required_streak:
                # advance curriculum: widen to 30, 60, 90, ... degrees
                curriculum_stage += 1
                theta_width_deg = 30.0 * curriculum_stage
                # cap at something sane like 180°
                theta_width_deg = min(theta_width_deg, 180.0)
                streak = 0

                # save weights for this band: 30/60/90 etc.
                save_deg = int(theta_width_deg)
                weight_path = f"Weights/Policy_Weights_{save_deg}deg.pth"
                torch.save(current_policy.state_dict(), weight_path)
                print(f">>> Curriculum widened to ±{theta_width_deg:.1f}°, "
                      f"weights saved to {weight_path} <<<")

            episode += 1

    # (unreachable unless you add a manual break somewhere above)
    torch.save(current_policy.state_dict(), "Weights/Policy_Weights_final.pth")
    print("Final weights saved to Weights/Policy_Weights_final.pth")
