import pathlib
import random
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from tqdm import tqdm

OUTPUT_PATH = pathlib.Path("outputs/plots/simulation")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


class GroupSimulation:
    def __init__(self, n_guardians_per_group, n_group_tasks, n_personal_nudges, tau):

        self.n_guardians_per_group = n_guardians_per_group
        self.guardian_list = list(range(n_guardians_per_group))
        self.n_group_tasks = n_group_tasks
        self.n_personal_nudges = n_personal_nudges

        self.tau = tau
        self.current_period = 0
        self.learning_period = 2
        self.eps = 1e-10

        self.data = []
        self.nudge_history = []

        # self.params is a dict that contains params for every guardian
        # e.g. guardian i's p parameter for k-th group task: params[i]['p'][k]
        self.params = {}

        for i in range(n_guardians_per_group):

            p = list(np.random.uniform(0, 1, self.n_group_tasks))
            q = list(np.random.uniform(0, 1, self.n_personal_nudges))

            self.params[i] = {}
            self.params[i]["p"] = p
            self.params[i]["q"] = q

    def intervene(self, intervention_function, periods):

        for _ in tqdm(range(periods)):

            group_task, personal_nudges = intervention_function(self)
            self.nudge_history.append((group_task, personal_nudges))

            for guardian in range(self.n_guardians_per_group):

                p = self.params[guardian]["p"][group_task]

                if guardian in personal_nudges.keys():
                    nudge = personal_nudges[guardian]
                    q = self.params[guardian]["q"][nudge]
                else:
                    nudge = None
                    q = 0

                y = bernoulli.rvs(p + (1 - p) * q)

                data = [self.current_period, guardian, group_task, nudge, y]

                self.data.append(data)

            self.current_period += 1

    def play_each_arm_once(self):

        """
        Method to observe each arm once, i.e. each individual intervention and each intervention,guardian,nudge possibility.
        Does not add to nudge history as it is not strategic nudge.
        """

        # total of L+L*K*N/tau periods

        # play all group tasks once without nudges
        for group_task in range(self.n_group_tasks):
            for guardian in range(self.n_guardians_per_group):

                y = bernoulli.rvs(self.params[guardian]["p"][group_task])

                self.data.append([self.current_period, guardian, group_task, None, y])

            # play each personal nudge under each group task for each guardian
            for nudge in range(self.n_personal_nudges):
                for guardian in range(self.n_guardians_per_group):

                    p = self.params[guardian]["p"][group_task]
                    q = self.params[guardian]["q"][nudge]
                    y = bernoulli.rvs(p + (1 - p) * q)

                    self.data.append([self.current_period, guardian, group_task, nudge, y])

        self.current_period += 1

    def get_df(self):

        df = pd.DataFrame(self.data)
        df.columns = ["period", "guardian", "group_task", "personalized_nudge", "y"]

        return df

    def get_optimal_nudges(self):

        L = self.n_group_tasks
        N = self.n_guardians_per_group

        opt_reward = 0

        p_params = [el["p"] for el in self.params.values()]
        task_rewards = np.sum(p_params, axis=0)

        for task in range(L):

            reward = 0

            task_reward = task_rewards[task]
            reward += task_reward

            a_is = []
            k_is = []

            for i in range(N):

                k_i = np.argmax(self.params[i]["q"])
                a_i = (1 - self.params[i]["p"][task]) * self.params[i]["q"][k_i]

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
                l_opt = task
                nudges_opt = nudges

        personal_nudges_opt = {}

        for guardian, nudge, _ in nudges_opt:

            personal_nudges_opt[guardian] = nudge

        return (opt_reward, [l_opt, task_rewards[l_opt]], personal_nudges_opt)

    def regret(self, group_task, personal_nudges):

        opt_reward, opt_group_task, opt_personal_nudges = self.get_optimal_nudges()

        task_reward = np.sum([el[group_task] for el in [param["p"] for param in self.params.values()]])

        total_reward = 0

        for guardian, nudge in personal_nudges.items():

            total_reward += (1 - self.params[guardian]["p"][group_task]) * self.params[guardian]["q"][nudge]

        total_reward += task_reward

        regret = opt_reward - total_reward

        if regret < self.eps:
            regret = 0

        return regret

    def cumulative_regret(self):

        regret_history = []
        cumulative = 0

        for task, nudges in self.nudge_history:

            regret = self.regret(task, nudges)
            cumulative += regret

            regret_history.append(cumulative)

        return regret_history


def random_intervention(gsim: GroupSimulation):

    """
    Outputs group task and personal nudges (dict with guardian indices j as keys, and nudge indices as values)
    """

    group_task = np.random.randint(0, gsim.n_group_tasks)

    guardians_nudged = random.sample(gsim.guardian_list, gsim.tau)
    personal_nudges = {}

    for guardian in guardians_nudged:
        personal_nudges[guardian] = np.random.randint(0, gsim.n_personal_nudges)

    return group_task, personal_nudges


def two_stage_intervention(gsim: GroupSimulation):

    """
    Obs: this intervention requires that some previous interventions were made and data is stored in self.data
    Obs2: this algorithm assumes knowledge of p's and q's
    """

    params = gsim.params

    group_task_probs = [el["p"] for el in params.values()]
    sum_group_task_probs = np.sum(group_task_probs, axis=0)

    group_task_chosen = np.argmax(sum_group_task_probs)

    # list to store guardian a_ik values
    guardian_nudge_scores = []

    for guardian in range(gsim.n_guardians_per_group):

        p = params[guardian]["p"][group_task_chosen]

        nudge = np.argmax(params[guardian]["q"])
        q = params[guardian]["q"][nudge]

        score = (1 - p) * q

        guardian_nudge_scores.append((guardian, nudge, score))

    # sort guardians from highest to lowest a_ik score
    guardian_values = sorted(guardian_nudge_scores, key=lambda x: x[2], reverse=True)

    # choose nudged guardians: tau greatest
    nudged_guardians = guardian_values[: gsim.tau]

    personal_nudges = {}

    for guardian, nudge, _ in nudged_guardians:
        personal_nudges[guardian] = nudge

    return group_task_chosen, personal_nudges


def two_stage_UCB_intervention(gsim: GroupSimulation, delta_func):

    df = gsim.get_df()
    delta = delta_func(gsim.current_period)

    # matrix with guardian and group task as index that maps to mean response of guardian in periods
    # where group task is applied and no personal nudge is applied to the guardian
    # e.g. p_matrix[i][j] = mean response (y) of guardian i under only group task j
    p_matrix = df[df["personalized_nudge"].isna()].groupby(["guardian", "group_task"])["y"].mean()

    # same thing but with the number of periods where group task is applied and no personal nudge is applied to the guardian
    n_matrix = df[df["personalized_nudge"].isna()].groupby(["guardian", "group_task"])["y"].size()

    # create matrix of upper confidence bound = p + sqrt(log(1/delta)/2n) and lower confidence bound
    UCB = np.sqrt(np.log(1 / delta) / (2 * n_matrix))
    UCB += p_matrix

    # if some ucb>1, lower to 1
    UCB = np.minimum(UCB, 1)

    LCB = np.sqrt(np.log(1 / delta) / (2 * n_matrix))
    LCB = p_matrix - LCB

    # if some lcb < 0, make it 0
    LCB = np.maximum(LCB, 0)

    # sum of each tasks ucb's over guardians
    sum_UCB = UCB.reset_index().groupby("group_task")["y"].sum()

    group_task = sum_UCB.argmax()

    # empty dict that receives guardians as keys and another dict as value
    # inner dict maps personalized nudges to their respective a_ik scores
    a_estimates = {}

    for guardian in gsim.guardian_list:

        a_estimates[guardian] = {}

        for nudge in range(gsim.n_personal_nudges):

            # for each personalized nudge, find the group task with most joint observations up until that time
            df_lk = df[(df["guardian"] == guardian) & (df["personalized_nudge"] == nudge)]

            l_k = df_lk["group_task"].value_counts().idxmax()

            mean_response = df_lk[df_lk["group_task"] == l_k]["y"].mean()
            n_ilk = df_lk[df_lk["group_task"] == l_k].shape[0]

            # for numeric reasons, if hat(p_ilk)=1, make it 1-eps
            # if p_matrix[guardian][l_k] == 1:
            #    p_ilk = p_matrix[guardian][l_k] - gsim.eps
            # else:
            #    p_ilk = p_matrix[guardian][l_k]

            # q_ik = (mean_response - p_ilk)/(1-p_ilk)
            a_ik = mean_response + np.sqrt(np.log(1 / delta) / (2 * n_ilk)) - LCB[guardian][l_k]

            a_estimates[guardian][nudge] = a_ik

    # for each guardian, calculate maximum a_ik score
    max_scores = []

    for guardian in gsim.guardian_list:

        a_scores = a_estimates[guardian]
        nudge = max(a_scores, key=lambda k: a_scores[k])
        score = a_scores[nudge]

        max_scores.append((guardian, nudge, score))

    sorted_scores = sorted(max_scores, key=lambda x: x[2])
    nudged_guardians = sorted_scores[-gsim.tau :]

    personal_nudges = {}

    for guardian, nudge, _ in nudged_guardians:
        personal_nudges[guardian] = nudge

    return group_task, personal_nudges


def full_UCB_intervention(gsim: GroupSimulation, delta_func):

    df = gsim.get_df()
    delta = delta_func(gsim.current_period)

    # matrix with guardian and group task as index that maps to mean response of guardian in periods
    # where group task is applied and no personal nudge is applied to the guardian
    # e.g. p_matrix[i][j] = mean response (y) of guardian i under only group task j
    p_matrix = df[df["personalized_nudge"].isna()].groupby(["guardian", "group_task"])["y"].mean()

    # same thing but with the number of periods where group task is applied and no personal nudge is applied to the guardian
    n_matrix = df[df["personalized_nudge"].isna()].groupby(["guardian", "group_task"])["y"].size()

    # create matrix of upper confidence bound = p + sqrt(log(1/delta)/2n) and lower confidence bound
    UCB = np.sqrt(np.log(1 / delta) / (2 * n_matrix))
    UCB += p_matrix

    # if some ucb>1, lower to 1
    # UCB = np.minimum(UCB, 1)

    LCB = np.sqrt(np.log(1 / delta) / (2 * n_matrix))
    LCB = p_matrix - LCB

    # if some lcb < 0, make it 0
    # LCB = np.maximum(LCB, 0)

    # sum of each tasks ucb's over guardians
    sum_UCB = UCB.reset_index().groupby("group_task")["y"].sum()

    # empty list that receives group tasks and their personal nudge strategy
    optimal_nudges_by_group_task = []

    # for each guardian/nudge and group task pair, calculate ucb of estimated reward

    for group_task in range(gsim.n_group_tasks):

        total_reward = 0

        task_reward = sum_UCB[group_task]
        total_reward += task_reward

        df_l = df[df["group_task"] == group_task]

        # fixed a group task, we choose nudges that max q_ik
        guardian_nudges = []

        for guardian in range(gsim.n_guardians_per_group):

            # find optimal nudge for each guardian under group task l

            nudge_rewards = []

            for nudge in range(gsim.n_personal_nudges):

                df_ilk = df_l[(df_l["guardian"] == guardian) & (df_l["personalized_nudge"] == nudge)]

                y_mean_ilk = df_ilk["y"].mean()
                n_ilk = df_ilk.shape[0]

                score = y_mean_ilk + np.sqrt((0.5 / n_ilk) * np.log(1 / delta)) - LCB[guardian][group_task]
                nudge_rewards.append((nudge, score))

            best_nudge = max(nudge_rewards, key=lambda x: x[1])

            guardian_nudges.append((guardian, best_nudge[0], best_nudge[1]))

        # sort guardians by score
        guardian_nudges = sorted(guardian_nudges, key=lambda x: x[2])

        # list of guardians to be nudged
        guardians_nudged = guardian_nudges[-gsim.tau :]

        personal_nudges = {}

        for guardian, nudge, _ in guardians_nudged:

            personal_nudges[guardian] = nudge

        # get final reward of optimal strategy under group task l
        nudge_reward = sum([x[2] for x in guardians_nudged])
        total_reward += nudge_reward

        optimal_nudges_by_group_task.append((group_task, personal_nudges, total_reward))

    # get final nudges
    group_task, personal_nudges = max(optimal_nudges_by_group_task, key=lambda x: x[2])[:2]

    return (group_task, personal_nudges)


##############################################################################################

n_guardians_per_group = 10
n_group_tasks = 2
n_personal_nudges = 2
tau = 2
random_periods = 400
twostage_periods = 200


def delta_func(t):
    return 1 / t


g = GroupSimulation(n_guardians_per_group, n_group_tasks, n_personal_nudges, tau)
g.get_optimal_nudges()
# g.intervene(random_intervention, random_periods)
g.play_each_arm_once()

g.intervene(random_intervention, 50)
func = partial(full_UCB_intervention, delta_func=delta_func)
g.intervene(func, 2000)

regret_timeline = g.cumulative_regret()
plt.plot(regret_timeline)
plt.savefig(OUTPUT_PATH / "regret.png")
plt.close()
breakpoint()

df = pd.DataFrame(g.data)
df.columns = ["period", "guardian", "group_task", "personalized_nudge", "y"]

df.groupby("period")["y"].sum()[1:].plot()
plt.savefig(OUTPUT_PATH / "optimal_algo.png")
breakpoint()
