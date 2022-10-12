import argparse
import itertools
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--group_id",
    "-g",
    default=29782,
    help="Group id to do analysis",
    type=int,
)
group.add_argument(
    "--random_groups",
    "-rg",
    help="Number of random group ids to do analysis",
    type=int,
)

args = parser.parse_args()

DATA_PATH = pathlib.Path("data/processed")
DATABASES_PATH = pathlib.Path("outputs/databases")

OUTPUT_PATH = pathlib.Path("outputs/plots/heatmaps")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = general_utils.open_pickle(DATABASES_PATH / "daily_features_database.pickle")

group_ids = df["group_id"].dropna().unique()

if args.random_groups:
    selected_group_ids = random.sample(list(group_ids), args.random_groups)
elif args.group_id:
    selected_group_ids = [args.group_id]
else:
    raise ValueError("Group choice method not implemented yet")

intervention_types = df.columns[4:12].tolist()

for group in selected_group_ids:

    dfg = df[df["group_id"] == group]

    guardian_ids = sorted(set(itertools.chain.from_iterable(dfg["guardian_ids"])))

    group_matrix = np.zeros([len(intervention_types), len(guardian_ids)])

    for i, itype in enumerate(intervention_types):

        df_active_intervention = dfg[dfg[itype]]
        n_days_under_intervention = df_active_intervention.shape[0]

        for j, gid in enumerate(guardian_ids):

            filter = df_active_intervention["guardian_ids"].explode() == gid
            df_guardian = df_active_intervention.loc[filter[filter].index]

            n_days_active_guardian_under_intervention = df_guardian.shape[0]

            try:
                prob_itype_gid = n_days_active_guardian_under_intervention / n_days_under_intervention
                group_matrix[i, j] = prob_itype_gid
            except ZeroDivisionError:
                group_matrix[i, j] = None

    group_matrix = np.round(group_matrix, 2)

    if group_matrix.shape[1] == 0:
        print(f"Group {group} has no observations!")
        continue
    elif np.sum(group_matrix) == 0:
        print(f"Group {group} has only null observations!")
        continue
    else:
        pass

    # fill na with 0
    group_matrix[np.isnan(group_matrix)] = 0

    # plot

    try:
        if group_matrix.shape[1] < 50:
            xsize = 15
            if group_matrix.shape[1] < 20:
                ysize = 6.5
            else:
                ysize = 5
        else:
            xsize, ysize = 25, 10

        fig, ax = plt.subplots(figsize=(xsize, ysize))

        sns.heatmap(group_matrix, center=0, cmap="vlag", linewidths=0.75, ax=ax, square=True, vmin=0)

        ax.set_xticklabels(guardian_ids)
        ax.set_yticklabels(intervention_types)

        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        plt.title(f"Group {group} - Prob. of interaction given guardian and intervention")

        fig.tight_layout()

        plt.savefig(OUTPUT_PATH / f"group_{group}-heatmap.png")

        plt.clf()
        plt.close()

    except ValueError:
        breakpoint()

    time.sleep(0.1)
