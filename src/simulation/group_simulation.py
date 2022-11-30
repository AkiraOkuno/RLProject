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

    def get_df(self):

        df = pd.DataFrame(self.data)
        df.columns = ["period", "guardian", "group_task", "personalized_nudge", "y"]

        return df

    def cumulative_regret(self):
        pass


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


def two_stage_UCB_intervention(gsim: GroupSimulation, delta):

    params = gsim.params
    df = gsim.get_df()

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
            if p_matrix[guardian][l_k] == 1:
                p_ilk = p_matrix[guardian][l_k] - gsim.eps
            else:
                p_ilk = p_matrix[guardian][l_k]

            # q_ik = (mean_response - p_ilk)/(1-p_ilk)
            a_ik = mean_response + np.sqrt(np.log(1 / delta) / (2 * n_ilk)) - LCB[guardian][group_task]

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


##############################################################################################

n_guardians_per_group = 20
n_group_tasks = 5
n_personal_nudges = 5
tau = 5
delta = 0.05
random_periods = 100
twostage_periods = 200

g = GroupSimulation(n_guardians_per_group, n_group_tasks, n_personal_nudges, tau)
g.intervene(random_intervention, random_periods)

func = partial(two_stage_UCB_intervention, delta=delta)
g.intervene(func, twostage_periods)

df = pd.DataFrame(g.data)
df.columns = ["period", "guardian", "group_task", "personalized_nudge", "y"]

df.groupby("period")["y"].sum().plot()
plt.savefig(OUTPUT_PATH / f"optimal_algo.png")
breakpoint()
