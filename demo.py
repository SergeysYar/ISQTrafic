"""Console demonstration of the traffic-control MVP."""

import argparse
import csv
import os
import random
import time

import numpy as np

from agent import QLearningAgent
from forecast import moving_average_forecast
from simulator import IntersectionSimulator


def longest_queue_action(state):
    return 0 if state[0] >= state[1] else 1


def train_agent(episodes, steps, seed):
    random.seed(seed)
    np.random.seed(seed)
    sim = IntersectionSimulator(arrival_rates=(0.6, 0.4), seed=seed)
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.3)
    for episode in range(episodes):
        state = sim.reset()
        epsilon = max(0.02, 0.3 * (1 - episode / max(1, episodes)))
        for _ in range(steps):
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, _, _ = sim.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
    return agent


def run_controller(controller, episodes, steps, seed):
    sim = IntersectionSimulator(arrival_rates=(0.6, 0.4), seed=seed)
    rewards, queues, discharged = [], [], []
    for _ in range(episodes):
        state = sim.reset()
        episode_reward = 0.0
        episode_queue = 0.0
        episode_discharged = 0
        for _ in range(steps):
            action = controller(state)
            state, reward, _, info = sim.step(action)
            episode_reward += reward
            episode_queue += sum(state)
            episode_discharged += info["discharged"]
        rewards.append(episode_reward)
        queues.append(episode_queue / steps)
        discharged.append(episode_discharged)
    return {
        "avg_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "avg_queue": float(np.mean(queues)),
        "avg_discharged": float(np.mean(discharged)),
    }


def print_live_episode(agent, steps, seed, delay):
    sim = IntersectionSimulator(arrival_rates=(0.6, 0.4), seed=seed)
    state = sim.reset()
    history = []
    print("\nПошаговая демонстрация Q-learning:")
    print("Легенда: NS — север/юг, EW — восток/запад, зелёный — выбранное направление")
    for step in range(1, steps + 1):
        action = agent.act(state, epsilon=0.0)
        next_state, reward, _, info = sim.step(action)
        history.append(sum(next_state))
        forecast = moving_average_forecast(history, window=5)
        direction = "NS" if action == 0 else "EW"
        bars = "NS [" + "#" * min(next_state[0], 30) + "] " + str(next_state[0])
        bars += " | EW [" + "#" * min(next_state[1], 30) + "] " + str(next_state[1])
        print(
            f"  t={step:02d} зелёный={direction} прибыло={sum(info['arrivals']):2d} "
            f"обслужено={info['discharged']:1d} reward={reward:3d} прогноз очереди={forecast:5.2f}\n"
            f"       {bars}"
        )
        state = next_state
        if delay > 0:
            time.sleep(delay)


def save_metrics(path, q_metrics, rule_metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["controller", "avg_reward", "reward_std", "avg_queue", "avg_discharged"])
        writer.writerow(["q_learning", q_metrics["avg_reward"], q_metrics["reward_std"], q_metrics["avg_queue"], q_metrics["avg_discharged"]])
        writer.writerow(["longest_queue", rule_metrics["avg_reward"], rule_metrics["reward_std"], rule_metrics["avg_queue"], rule_metrics["avg_discharged"]])


def main():
    parser = argparse.ArgumentParser(description="Демонстрация управления светофором")
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--live-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.0, help="Пауза между шагами, например 0.2")
    parser.add_argument("--qtable", default="artifacts/demo_qtable.pkl")
    parser.add_argument("--metrics", default="artifacts/demo_metrics.csv")
    args = parser.parse_args()

    print("=== ISQTrafic: демонстрация MVP ===")
    print(f"Обучение: {args.train_episodes} эпизодов по {args.train_steps} шагов")
    agent = train_agent(args.train_episodes, args.train_steps, args.seed)
    os.makedirs(os.path.dirname(args.qtable) or ".", exist_ok=True)
    agent.save(args.qtable)
    q_metrics = run_controller(lambda state: agent.act(state, epsilon=0.0), args.eval_episodes, args.eval_steps, args.seed + 1)
    rule_metrics = run_controller(longest_queue_action, args.eval_episodes, args.eval_steps, args.seed + 1)
    save_metrics(args.metrics, q_metrics, rule_metrics)
    print("\nРезультаты на одинаковых случайных потоках:")
    print("Контроллер       Награда       Средняя очередь   Обслужено")
    print(f"Q-learning       {q_metrics['avg_reward']:8.2f}       {q_metrics['avg_queue']:8.2f}         {q_metrics['avg_discharged']:6.2f}")
    print(f"Длинная очередь  {rule_metrics['avg_reward']:8.2f}       {rule_metrics['avg_queue']:8.2f}         {rule_metrics['avg_discharged']:6.2f}")
    print(f"\nQ-table сохранена: {args.qtable}")
    print(f"Метрики сохранены: {args.metrics}")
    print_live_episode(agent, args.live_steps, args.seed + 2, args.delay)


if __name__ == "__main__":
    main()
