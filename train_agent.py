import argparse
import os
import numpy as np

from simulator import IntersectionSimulator
from agent import QLearningAgent


def train(episodes=200, steps_per_episode=200, out_path="qtable.pkl"):
    sim = IntersectionSimulator(arrival_rates=(0.6, 0.4), seed=0)
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.3)

    for ep in range(episodes):
        s = sim.reset()
        total = 0.0
        eps = max(0.02, 0.3 * (1 - ep / episodes))
        for _ in range(steps_per_episode):
            a = agent.act(s, epsilon=eps)
            ns, r, d, _ = sim.step(a)
            agent.update(s, a, r, ns)
            s = ns
            total += r
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} reward={total:.1f} epsilon={eps:.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    agent.save(out_path)
    print(f"Saved Q-table to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=str, default="qtable.pkl")
    args = parser.parse_args()
    train(episodes=args.episodes, steps_per_episode=args.steps, out_path=args.out)
