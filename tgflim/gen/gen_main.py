import numpy as np
import matplotlib.pyplot as plt
from configs import Config


class Generator:
    def __init__(self, config: Config):
        self.config = config

    def genTrace(self):
        numgates = int(self.config.freq * self.config.integ)
        prob = np.zeros(len(self.config.times))
        for i, lt in enumerate(self.config.lifetimes):
            lam = 1.0 / lt
            prob += self.config.weight[i] * self.config.zeta * (
                np.exp(-lam * self.config.times) -
                np.exp(-lam * (self.config.times + self.config.width))
            )
        counts = np.random.binomial(numgates, np.clip(prob, 0, 1))
        return counts

    def genImage(self):
        img = np.zeros((self.config.numsteps, self.config.x, self.config.y), dtype=int)
        for i in range(self.config.x):
            for j in range(self.config.y):
                img[:, i, j] = self.genTrace()
        return img
