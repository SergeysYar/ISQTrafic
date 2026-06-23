import argparse
import numpy as np
from simulator import IntersectionSimulator
from agent import QLearningAgent


def evaluate(agent_path, episodes=100, steps=200):
    sim = IntersectionSimulator(arrival_rates=(0.6, 0.4))
    agent = QLearningAgent()
    agent.load(agent_path)

    totals = []
    for _ in range(episodes):
        s = sim.reset()
        total = 0.0
        for _ in range(steps):
            a = agent.act(s, epsilon=0.0)
            s, r, d, _ = sim.step(a)
            total += r
        totals.append(total)
    print(f"Avg reward: {np.mean(totals):.2f} +- {np.std(totals):.2f}")


if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.agent, episodes=args.episodes, steps=args.steps)
