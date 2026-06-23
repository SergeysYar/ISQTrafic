import numpy as np
import pickle


class QLearningAgent:
    """Tabular Q-learning agent for two-action traffic-control environment."""

    def __init__(self, actions=(0, 1), alpha=0.1, gamma=0.99, epsilon=0.2):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}  # state -> np.array(len(actions))

    def _ensure_state(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(len(self.actions), dtype=float)

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        self._ensure_state(state)
        if np.random.rand() < epsilon:
            return int(np.random.choice(self.actions))
        else:
            return int(self.actions[int(np.argmax(self.q[state]))])

    def update(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)
        a_idx = self.actions.index(action)
        best_next = np.max(self.q[next_state])
        td = reward + self.gamma * best_next - self.q[state][a_idx]
        self.q[state][a_idx] += self.alpha * td

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.q = pickle.load(f)


if __name__ == "__main__":
    from simulator import IntersectionSimulator
    sim = IntersectionSimulator(seed=0)
    agent = QLearningAgent()
    s = sim.reset()
    for _ in range(20):
        a = agent.act(s)
        ns, r, d, _ = sim.step(a)
        agent.update(s, a, r, ns)
        s = ns
        sim.render()