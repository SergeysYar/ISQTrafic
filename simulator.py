import numpy as np


class IntersectionSimulator:
    """Простейший симулятор перекрёстка с двумя направлениями: NS и EW.

    State: tuple (queue_ns, queue_ew) — целые длины очередей.
    Action: 0 = дать зелёный NS, 1 = дать зелёный EW.
    """

    def __init__(self, arrival_rates=(0.5, 0.5), max_queue=50, seed=None):
        self.arrival_rates = tuple(arrival_rates)
        self.max_queue = max_queue
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.queues = np.array([0, 0], dtype=int)
        self.t = 0
        return self._get_state()

    def step(self, action):
        """Execute one timestep.

        Returns: (next_state, reward, done, info)
        """
        # New arrivals (Poisson)
        arrivals = self.rng.poisson(self.arrival_rates)
        self.queues += arrivals

        # Service: green direction discharges up to capacity
        capacity = 2
        if action == 0:
            discharged = min(self.queues[0], capacity)
            self.queues[0] -= discharged
        else:
            discharged = min(self.queues[1], capacity)
            self.queues[1] -= discharged

        # Clip queues to max
        self.queues = np.clip(self.queues, 0, self.max_queue)

        self.t += 1
        reward = -int(self.queues.sum())
        done = False
        info = {"discharged": int(discharged), "arrivals": arrivals.tolist()}
        return self._get_state(), reward, done, info

    def _get_state(self):
        return (int(self.queues[0]), int(self.queues[1]))

    def render(self):
        print(f"t={self.t} NS={self.queues[0]} EW={self.queues[1]}")


if __name__ == "__main__":
    sim = IntersectionSimulator(arrival_rates=(0.6, 0.4), seed=1)
    s = sim.reset()
    for _ in range(10):
        a = 0 if sim.queues[0] >= sim.queues[1] else 1
        s, r, d, info = sim.step(a)
        sim.render()
