import argparse
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

DATABASES_PATH = pathlib.Path("outputs/databases")
OUTPUT_PATH = pathlib.Path("outputs/models")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--remove_2022",
    "-r",
    action="store_true",
    help="Whether to remove entries in 2022 from analysis",
)

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

df_daily = general_utils.open_pickle(DATABASES_PATH / "daily_features_database.pickle")

# Filter only 2021 because intervention data for 2022 is incomplete
if args.remove_2022:
    df_daily = df_daily[df_daily["sent_day"].dt.year == 2021]

group_ids = df_daily["group_id"].dropna().unique()

if args.random_groups:
    selected_group_ids = random.sample(list(group_ids), args.random_groups)
elif args.group_id:
    selected_group_ids = [args.group_id]
else:
    raise ValueError("Group choice method not implemented yet")

for gid in tqdm(selected_group_ids):

    dfg = df_daily[df_daily["group_id"] == gid]

    # skip group if too few observations
    if dfg.shape[0] < 100:
        continue

    # daily feature: number of messages in that day
    dfg["daily_total_messages"] = dfg["guardian_ids_history"].apply(len)

    # calculate hour dummies and discretize by 3 hour intervals
    df_hour = pd.get_dummies(dfg["DA_intervention_hours"].apply(pd.Series).stack()).sum(level=0)

    for h in range(23):
        if h not in df_hour.columns:
            df_hour[h] = 0

    for h in range(8):
        col = f"{3*h}-{3*h+2}"
        dfg[col] = 0

        for i in range(3):

            if df_hour.shape[0] > 0:

                dfg[col] += df_hour[3 * h + i]

                # fill nas with 0, i.e. all hour values are 0 for days without interventions
                dfg[col] = dfg[col].fillna(0)

            else:
                # if there is no intervention in the whole period
                dfg[col] = 0

        # map non zeroes to 1
        dfg[col] = dfg[col].astype(bool).astype(int)

    intervention_values = dfg[dfg.columns[5:14]].sum().values
    top5_interventions = dfg.columns[5:14][intervention_values.argsort()[-5:][::-1]]
    remove = [col for col in dfg.columns[5:14] if col not in top5_interventions]

    # daily outcome regression
    y = dfg["n_guardians"]

    X = dfg.drop(columns=remove)
    X = X.drop(
        columns=[
            "n_guardians",
            "group_id",
            "sent_day",
            "guardian_ids",
            "guardian_ids_history",
            "DA_intervention_hours",
            "weekday",
            "0-2",
        ]
    )

    for col in X.columns:
        X[col] = X[col].astype(int)

    # delete possible only 0 columns
    X = X.loc[:, (X != 0).any(axis=0)]

    # add constant to model
    X["constant"] = 1

    reg = sm.OLS(y, X).fit()

    with open(OUTPUT_PATH / "daily_outcome_model.txt", "a+") as f:  # a+
        f.write(f"Group {gid} regression results:\n\n")
        f.write(reg.summary().as_text())
        f.write("\n==============================================================================\n\n")

    time.sleep(0.1)
