import random
from scipy.stats import bernoulli
import numpy as np
import scipy.optimize as opt
from tqdm import tqdm
import pathlib
from matplotlib import pyplot as plt

OUTPUT_PATH = pathlib.Path("outputs/plots/simulation")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


class Simulation:
    def __init__(self, n_group_tasks, n_personal_nudges, p, q):

        assert len(p) == n_group_tasks
        assert len(q) == n_personal_nudges

        self.n_group_tasks = n_group_tasks
        self.n_personal_nudges = n_personal_nudges
        self.p = p

        # add prob=0 to q_list == no increased prob. of response if no nudge is applied

        self.q = q.copy()
        self.q.insert(0, 0)

        self.data = []

    def simulate_data(self, t):

        for _ in range(t):

            group_task = random.choice(range(self.n_group_tasks))

            # nudge = 0 => no nudge is applied
            nudge = random.choice(range(self.n_personal_nudges + 1))

            prob = self.p[group_task] + (1 - self.p[group_task]) * self.q[nudge]

            y = bernoulli.rvs(prob)

            self.data.append([group_task, nudge, y])

        self.data = np.array(self.data)

    # for now, i will assume that K=L=2, and it remains as TODO to make a generalized function
    def calculate_loglikelihood_coefficients(self):

        # nl0's
        n00 = self.data[(self.data[:, 0] == 0) & (self.data[:, 1] == 0)].shape[0]
        n10 = self.data[(self.data[:, 0] == 1) & (self.data[:, 1] == 0)].shape[0]
        sum_y00 = self.data[(self.data[:, 0] == 0) & (self.data[:, 1] == 0)][:, 2].sum()
        sum_y10 = self.data[(self.data[:, 0] == 1) & (self.data[:, 1] == 0)][:, 2].sum()

        # nlk's
        n01 = self.data[(self.data[:, 0] == 0) & (self.data[:, 1] == 1)].shape[0]
        n02 = self.data[(self.data[:, 0] == 0) & (self.data[:, 1] == 2)].shape[0]
        n11 = self.data[(self.data[:, 0] == 1) & (self.data[:, 1] == 1)].shape[0]
        n12 = self.data[(self.data[:, 0] == 1) & (self.data[:, 1] == 2)].shape[0]

        sum_y01 = self.data[(self.data[:, 0] == 0) & (self.data[:, 1] == 1)][:, 2].sum()
        sum_y02 = self.data[(self.data[:, 0] == 0) & (self.data[:, 1] == 2)][:, 2].sum()
        sum_y11 = self.data[(self.data[:, 0] == 1) & (self.data[:, 1] == 1)][:, 2].sum()
        sum_y12 = self.data[(self.data[:, 0] == 1) & (self.data[:, 1] == 2)][:, 2].sum()

        self.coefficients = [n00, n10, sum_y00, sum_y10, n01, n02, n11, n12, sum_y01, sum_y02, sum_y11, sum_y12]

    def get_loglikelihood(self):

        n00, n10, sum_y00, sum_y10, n01, n02, n11, n12, sum_y01, sum_y02, sum_y11, sum_y12 = self.coefficients

        def loglik2(parameters):

            p1, p2, q1, q2 = parameters
            val = (
                sum_y00 * np.log((p1 / (1 - p1)))
                + n00 * np.log(1 - p1)
                + sum_y01 * np.log((p1 + (1 - p1) * q1) / (1 - p1 - (1 - p1) * q1))
                + n01 * np.log(1 - p1 - (1 - p1) * q1)
                + sum_y02 * np.log((p1 + (1 - p1) * q2) / (1 - p1 - (1 - p1) * q2))
                + n02 * np.log(1 - p1 - (1 - p1) * q2)
                + sum_y10 * np.log((p2 / (1 - p2)))
                + n10 * np.log(1 - p2)
                + sum_y11 * np.log((p2 + (1 - p2) * q1) / (1 - p2 - (1 - p2) * q1))
                + n11 * np.log(1 - p2 - (1 - p2) * q1)
                + sum_y12 * np.log((p2 + (1 - p2) * q2) / (1 - p2 - (1 - p2) * q2))
                + n12 * np.log(1 - p2 - (1 - p2) * q2)
            )
            return -val

        def negloglik(parameters):

            p1, p2, q1, q2 = parameters

            val = 0

            # l=0,k=0
            val += sum_y00 * np.log((p1 / (1 - p1)))
            val += n00 * np.log(1 - p1)

            # l=0,k=1,2
            val += sum_y01 * np.log((p1 + (1 - p1) * q1) / (1 - p1 - (1 - p1) * q1))
            val += n01 * np.log(1 - p1 - (1 - p1) * q1)
            val += sum_y02 * np.log((p1 + (1 - p1) * q2) / (1 - p1 - (1 - p1) * q2))
            val += n02 * np.log(1 - p1 - (1 - p1) * q2)

            # l=1,k=0
            val += sum_y10 * np.log((p2 / (1 - p2)))
            val += n10 * np.log(1 - p2)

            # l=1,k=1,2
            val += sum_y11 * np.log((p2 + (1 - p2) * q1) / (1 - p2 - (1 - p2) * q1))
            val += n11 * np.log(1 - p2 - (1 - p2) * q1)
            val += sum_y12 * np.log((p2 + (1 - p2) * q2) / (1 - p2 - (1 - p2) * q2))
            val += n12 * np.log(1 - p2 - (1 - p2) * q2)

            return -val

        return negloglik


n_group_tasks = 2
n_personal_nudges = 2
p = [0.4, 0.1]
q = [0.8, 0.3]
theta = p + q
n_data = 100

n_simulations = 20000
EPS = 0.0001
bounds = ((0 + EPS, 1 - EPS), (0 + EPS, 1 - EPS), (0 + EPS, 1 - EPS), (0 + EPS, 1 - EPS))
soln_list = []

for _ in tqdm(range(n_simulations)):

    sim = Simulation(n_group_tasks, n_personal_nudges, p, q)
    sim.simulate_data(n_data)
    sim.calculate_loglikelihood_coefficients()
    negloglik = sim.get_loglikelihood()

    optim_object = opt.minimize(negloglik, x0=[0.5] * 4, bounds=bounds, method="L-BFGS-B")
    soln = optim_object["x"]
    soln_list.append(soln)

soln_list = np.array(soln_list)

# true value = black line
# mean estimate = red line
plt.hist(soln_list[:, 0], color="c", edgecolor="k", alpha=0.65, bins=25)
plt.axvline(theta[0], color="k", linestyle="dashed", linewidth=1)
plt.axvline(soln_list[:, 0].mean(), color="r", linestyle="dashed", linewidth=1)
plt.savefig(OUTPUT_PATH / f"p1_histogram.png")
plt.close()

plt.hist(soln_list[:, 1], color="c", edgecolor="k", alpha=0.65, bins=25)
plt.axvline(theta[1], color="k", linestyle="dashed", linewidth=1)
plt.axvline(soln_list[:, 1].mean(), color="r", linestyle="dashed", linewidth=1)
plt.savefig(OUTPUT_PATH / f"p2_histogram.png")
plt.close()

plt.hist(soln_list[:, 2], color="c", edgecolor="k", alpha=0.65, bins=25)
plt.axvline(theta[2], color="k", linestyle="dashed", linewidth=1)
plt.axvline(soln_list[:, 2].mean(), color="r", linestyle="dashed", linewidth=1)
plt.savefig(OUTPUT_PATH / f"q1_histogram.png")
plt.close()

plt.hist(soln_list[:, 3], color="c", edgecolor="k", alpha=0.65, bins=25)
plt.axvline(theta[3], color="k", linestyle="dashed", linewidth=1)
plt.axvline(soln_list[:, 3].mean(), color="r", linestyle="dashed", linewidth=1)
plt.savefig(OUTPUT_PATH / f"q2_histogram.png")
plt.close()

breakpoint()
