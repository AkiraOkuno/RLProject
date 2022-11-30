import numpy as np
import random
from scipy.stats import bernoulli
import pandas as pd
import pathlib
from matplotlib import pyplot as plt
from tqdm import tqdm

OUTPUT_PATH = pathlib.Path("outputs/plots/simulation")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


class Groups:
    def __init__(self, n_groups, n_guardians_per_group, n_group_tasks, n_personal_nudges, tau):

        self.n_groups = n_groups
        self.n_guardians_per_group = n_guardians_per_group
        self.guardian_list = [(i, j) for i in range(n_groups) for j in range(n_guardians_per_group)]
        self.n_group_tasks = n_group_tasks
        self.n_personal_nudges = n_personal_nudges
        self.tau = tau
        self.current_period = 0
        self.learning_period = 2
        self.eps = 10e-5

        self.data = []
        self.groups = {}

        for i in range(n_groups):

            group = {}

            for j in range(n_guardians_per_group):

                p = np.random.uniform(0, 1, n_group_tasks)
                q = np.random.uniform(0, 1, n_personal_nudges)

                group[j] = (p, q)

            self.groups[i] = {"guardian_parameters": group}

    def two_stage_intervention_learning(self):

        df = pd.DataFrame(self.data)
        df.columns = ["period", "guardian", "group_task", "personalized_nudge", "y"]

        # initialize empty objects for group tasks and personal nudges
        group_tasks = []
        personal_nudges = dict.fromkeys(self.guardian_list)

        # matrix with guardian and group task as index that maps to mean response of guardian in periods
        # where group task is applied and no personal nudge is applied to the guardian
        p_matrix = df[df["personalized_nudge"].isna()].groupby(["guardian", "group_task"])["y"].mean()

        # same thing but with the number of periods where group task is applied and no personal nudge is applied to the guardian
        p_T_matrix = df[df["personalized_nudge"].isna()].groupby(["guardian", "group_task"])["y"].size()

        # create matrix of upper confidence bound = p + sqrt(2*log(t)/p_T)
        UCB = np.sqrt(2 * np.log(self.learning_period) / p_T_matrix)
        UCB += p_matrix

        # if some ucb>1, transform to 1
        UCB = np.minimum(UCB, 1)

        # empty list that receives triples: guardian, nudge, q_ik
        q_estimates = []

        # calculate q values (one value for each guardian, personal nudge pair)
        for guardian in self.guardian_list:
            for nudge in range(self.n_personal_nudges):

                # get group task that had the most amount of data to calculate E[Y] in qik estimator
                q_group_task = (
                    df[(df["guardian"] == guardian) & (df["personalized_nudge"] == nudge)]
                    .groupby("group_task")["y"]
                    .size()
                    .idxmax()
                )

                # calculate E[Y] = mean response condition on guardian, group task and personal nudge
                EY = df[
                    (df["guardian"] == guardian)
                    & (df["personalized_nudge"] == nudge)
                    & (df["group_task"] == q_group_task)
                ]["y"].mean()

                # pij = mean response given guardian, group task and no nudge
                pij = p_matrix[guardian, q_group_task]

                if pij == 1:
                    pij -= self.eps

                # q estimator for guardian i and nudge k
                qik = (EY - pij) / (1 - pij)

                q_estimates.append([guardian, nudge, qik])

        # transform q estimates into dataframe
        q_estimates = pd.DataFrame(q_estimates)
        q_estimates.columns = ["guardian", "personalized_nudge", "q"]

        # normalize q's less than 0 to be equal to 0
        q_estimates["q"] = np.maximum(q_estimates["q"], 0)

        # for each group task, calculate sum of p's over all guardians in the group
        UCB = UCB.reset_index()
        UCB["group"] = UCB["guardian"].apply(lambda x: x[0])
        UCB_sum = UCB.groupby(["group", "group_task"])["y"].sum().reset_index()

        # now we calculate the effective nudges using the p UCB's and q estimates

        # for each group
        for i in range(self.n_groups):

            # get index of group task with highest p's UCBs sum
            idx = UCB_sum[UCB_sum["group"] == i]["y"].idxmax()

            # get chosen task code
            chosen_group_task = UCB_sum.loc[idx, "group_task"]

            # save group chosen task
            group_tasks.append(chosen_group_task)

            # initialize empty list for saving each (guardian, nudge) pair score = (1-p_i*)q_ik
            guardian_values = []

            # for each nudge
            for j in range(self.n_guardians_per_group):

                guardian = (i, j)

                # get p_i* = p_ij for group task j with highest ucb sum
                p = p_matrix[guardian, chosen_group_task]

                # get q_i* = prob.estimate q_ik for guardian i such that estimate q_ik is the highest over all nudges
                q_values = (
                    q_estimates[q_estimates["guardian"] == guardian]
                    .drop("guardian", axis=1)
                    .set_index("personalized_nudge")
                )
                q = q_values.max()[0]

                # get nudge number
                nudge = q_values.idxmax()[0]

                # calculate guardian score with group task and nudge
                guardian_values.append((guardian, nudge, (1 - p) * q))

            # sort guardian scores
            guardian_values = sorted(guardian_values, key=lambda x: x[2], reverse=True)

            # get the top tau guardians sorted by score
            nudged_guardians = guardian_values[: self.tau]

            # save personalized nudge value for each of the chosen guardians
            for guardian, nudge, _ in nudged_guardians:
                personal_nudges[guardian] = nudge

        # update t for next UCB estimate
        self.learning_period += 1

        return (group_tasks, personal_nudges)

    def two_stage_intervention(self):

        """
        Obs: this intervention requires that some previous interventions were made and data is stored in self.data
        Obs2: this algorithm assumes knowledge of p's and q's
        """

        group_tasks = []
        personal_nudges = dict.fromkeys(self.guardian_list)

        for i in range(self.n_groups):

            group_i = self.groups[i]["guardian_parameters"]
            group_task_probs = [el[0] for el in group_i.values()]
            sum_group_task_probs = np.sum(group_task_probs, axis=0)

            group_task_chosen = np.argmax(sum_group_task_probs)
            group_tasks.append(group_task_chosen)

            guardian_values = []

            for j in range(self.n_guardians_per_group):

                guardian = (i, j)
                p = group_i[j][0][group_task_chosen]

                nudge = np.argmax(group_i[j][1])
                q = group_i[j][1][nudge]

                prob = (1 - p) * q

                guardian_values.append((guardian, nudge, prob))

            guardian_values = sorted(guardian_values, key=lambda x: x[2], reverse=True)
            nudged_guardians = guardian_values[: self.tau]

            for guardian, nudge, _ in nudged_guardians:
                personal_nudges[guardian] = nudge

        return (group_tasks, personal_nudges)

    def random_intervention(self):

        """
        Output formats:
        - group tasks: list of size self.n_groups
        - personal nudges: dict with tuples (i,j) as keys, where i=group, j=guardian in group, and nudge integer as values

        """

        group_tasks = []

        for i in range(self.n_group_tasks):
            group_tasks.append(np.random.randint(0, self.n_group_tasks))

        personal_nudges = []

        personal_nudges = dict.fromkeys(self.guardian_list)

        guardians_nudged = random.sample(self.guardian_list, self.tau)

        for guardian in guardians_nudged:
            personal_nudges[guardian] = np.random.randint(0, self.n_personal_nudges)

        return (group_tasks, personal_nudges)

    def intervene(self, intervention_function, periods):

        for _ in tqdm(range(periods)):

            group_tasks, personal_nudges = intervention_function()

            for group in self.groups.keys():

                group_task = group_tasks[group]

                for i in range(self.n_guardians_per_group):

                    guardian = (group, i)
                    guardian_parameters = self.groups[group]["guardian_parameters"][i]

                    p = guardian_parameters[0][group_task]
                    nudge = personal_nudges[guardian]

                    if nudge is None:
                        q = 0
                    else:
                        q = guardian_parameters[1][nudge]

                    y = bernoulli.rvs(p + (1 - p) * q)

                    data = [self.current_period, guardian, group_task, nudge, y]

                    self.data.append(data)

            self.current_period += 1


n_groups = 3
n_guardians_per_group = 5
n_group_tasks = 3
n_personal_nudges = 4
tau = 2
random_periods = 200
twostage_periods = 100

g = Groups(n_groups, n_guardians_per_group, n_group_tasks, n_personal_nudges, tau)
g.intervene(g.random_intervention, random_periods)

# breakpoint()
g.intervene(g.two_stage_intervention, twostage_periods)
# g.intervene(g.two_stage_intervention_learning, twostage_periods)

df = pd.DataFrame(g.data)
df.columns = ["period", "guardian", "group_task", "personalized_nudge", "y"]

df.groupby("period")["y"].sum().plot()
plt.savefig(OUTPUT_PATH / f"optimal_algo.png")
