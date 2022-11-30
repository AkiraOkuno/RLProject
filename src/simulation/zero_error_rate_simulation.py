import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

EPS = 1e-10


def generate_param(N, L, K):

    params = []

    for i in range(N):

        p = list(np.random.uniform(0, 1, L))
        q = list(np.random.uniform(0, 1, K))

        params.append([p, q])

    return params


def algo_solution(params, tau, l=None):

    """
    Function that receives as input:
    - parameters array
    - tau (# nudged guardians) as input
    - l: (optional) if None, uses 2SIM optimal task, otherwise uses user-specified task
    and outputs:
    - reward of algorithmic 2SIM strategy
    - list_task: [optimal task, reward gained through this task]
    - list_nudges: list of triples (guardian, nudge code, nudge reward) for all tau nudged guardians
    """

    L = len(params[0][0])
    K = len(params[0][1])
    N = len(params)

    reward = 0

    task_rewards = [np.sum([p[0][l] for p in params]) for l in range(L)]

    if l is None:
        l_opt = np.argmax(task_rewards)
    else:
        l_opt = l

    task_reward = task_rewards[l_opt]

    reward += task_reward

    # fix l = l_opt

    a_is = []
    k_is = []

    for i in range(N):

        k_i = np.argmax(params[i][1])
        a_i = (1 - params[i][0][l_opt]) * params[i][1][k_i]

        k_is.append(k_i)
        a_is.append(a_i)

    nudges = []

    for i in range(1, tau + 1):

        ith_score = sorted(a_is)[-i]
        ith_guardian = a_is.index(ith_score)

        nudges.append([ith_guardian, k_is[ith_guardian], ith_score])
        reward += ith_score

    return (reward, [l_opt, task_reward], nudges)


def optimal_solution(params, tau):

    L = len(params[0][0])
    K = len(params[0][1])
    N = len(params)

    opt_reward = 0

    task_rewards = [np.sum([p[0][l] for p in params]) for l in range(L)]

    for l in range(L):

        reward = 0

        task_reward = task_rewards[l]
        reward += task_reward

        a_is = []
        k_is = []

        for i in range(N):

            k_i = np.argmax(params[i][1])
            a_i = (1 - params[i][0][l]) * params[i][1][k_i]

            k_is.append(k_i)
            a_is.append(a_i)

        nudges = []

        for i in range(1, tau + 1):

            ith_score = sorted(a_is)[-i]
            ith_guardian = a_is.index(ith_score)

            nudges.append([ith_guardian, k_is[ith_guardian], ith_score])
            reward += ith_score

        if reward > opt_reward:
            opt_reward = reward
            l_opt = l
            nudges_opt = nudges

    return (opt_reward, [l_opt, task_rewards[l_opt]], nudges_opt)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-N",
    help="Number of guardians",
    type=int,
    default=2,
)
parser.add_argument(
    "-L",
    help="Number of tasks",
    type=int,
    default=2,
)
parser.add_argument(
    "-K",
    help="Number of nudges for each guardian",
    type=int,
    default=2,
)
parser.add_argument(
    "-tau",
    help="Number of nudged guardians",
    type=int,
    default=1,
)
parser.add_argument(
    "-T",
    help="Number of times to simulate",
    type=int,
    default=500000,
)

args = parser.parse_args()

difference = []
params_list = []

zero_error_count = 0

for _ in tqdm(range(args.T)):

    params = generate_param(args.N, args.L, args.K)

    algo_reward, _, _ = algo_solution(params, args.tau)
    opt_reward, _, _ = optimal_solution(params, args.tau)

    # params = [[[0.4600921276313461, 0.8590847916799589], [0.6631699836316104, 0.9005348769021504]],[[0.8168015879150715, 0.4620709221520113], [0.7311153993107993, 0.041681119535392]]]
    # params = [[[0.7362225955593652, 0.32831492334751444], [0.04330147468984347, 0.1889122925303619]], [[0.02302785585133016, 0.5278124985463037], [0.42891852188121493, 0.694106447484428]]]

    delta = (opt_reward - algo_reward) / opt_reward
    difference.append(delta)

    if opt_reward - algo_reward < EPS:
        zero_error_count += 1

print(f"Zero error rate: {zero_error_count/args.T}")

breakpoint()

# params such that task 1 is chosen and task is effective to guardian 1
params1 = [
    x
    for x in params_list
    if x[0][0][0][0] + x[0][1][0][0] > x[0][0][0][1] + x[0][1][0][1] and x[0][0][0][0] > x[0][1][0][0]
]
# filter values where max(q11,q12)<max(q21,q22)
params1 = [x for x in params1 if max(x[0][0][1][0], x[0][0][1][1]) < max(x[0][1][1][0], x[0][1][1][1])]
# filter values where change of task would be more effective for guardian 1: p12>p11
params1 = [x for x in params1 if x[0][0][0][1] > x[0][0][0][0]]

error_params = [p for p in params_list if p[1] > 0.1]
sorted_params = sorted(error_params, key=lambda x: x[1], reverse=True)

# cases where task 1 is chosen and task is effective to guardian 1
sorted_params1 = [
    x
    for x in sorted_params
    if x[0][0][0][0] + x[0][1][0][0] > x[0][0][0][1] + x[0][1][0][1] and x[0][0][0][0] > x[0][1][0][0]
]

# breakpoint()

f, ax = plt.subplots()
points = ax.scatter(
    *zip(*[(x[0][0][0][0], x[0][1][0][0]) for x in sorted_params1]),
    c=[x[1] for x in sorted_params1],
    cmap="plasma",
    s=1,
)
f.colorbar(points)
plt.savefig("outputs/tests/test.png")
plt.close()

f, ax = plt.subplots()
points = ax.scatter(
    *zip(*[(x[0][0][0][1], x[0][1][0][1]) for x in sorted_params1]),
    c=[x[1] for x in sorted_params1],
    cmap="plasma",
    s=1,
)
f.colorbar(points)
plt.savefig("outputs/tests/test2.png")
plt.close()

# p11-p12 and p22-p21
f, ax = plt.subplots()
points = ax.scatter(
    *zip(*[(x[0][0][0][0] - x[0][0][0][1], x[0][1][0][1] - x[0][1][0][0]) for x in sorted_params1]),
    c=[x[1] for x in sorted_params1],
    cmap="plasma",
    s=1,
)
f.colorbar(points)
plt.savefig("outputs/tests/test3.png")
plt.close()

# max(q21,q22) and max(q11,q12) = maximum nudge effectivity for iL=2 (guardian that task is not effective) and for iH=1
f, ax = plt.subplots()
points = ax.scatter(
    *zip(*[(max(x[0][1][1][0], x[0][1][1][1]), max(x[0][0][1][0], x[0][0][1][1])) for x in sorted_params1]),
    c=[x[1] for x in sorted_params1],
    cmap="plasma",
    s=1,
)
f.colorbar(points)
plt.savefig("outputs/tests/test4.png")
plt.close()

plt.hist(difference, bins=50)
plt.savefig("outputs/tests/difference_hist.png")

print(f"Proportion of zero error: {np.sum(np.array(difference)>0.1)/T}")

breakpoint()

division_max_q_and_task1_division_p = [
    (max(p[1][1][0], p[1][1][1]) / max(p[0][1][0], p[0][1][1]), (1 - p[0][0][0]) / (1 - p[1][0][0]))
    for p in params_zero_list
]
# task1_division_p = [(1-p[0][0][0])/(1-p[1][0][0]) for p in params_zero_list]
# task2_division_p = [(1-p[0][0][0])/(1-p[0][0][1]) for p in params_zero_list]

plt.figure(figsize=(20, 20))
plt.scatter(*zip(*division_max_q_and_task1_division_p), s=0.1)
plt.savefig("outputs/tests/test.png")
plt.close()
