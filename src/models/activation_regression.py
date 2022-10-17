import argparse
import os
import pathlib
import random
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.append(os.getcwd())
from src.utils import general_utils

DATABASES_PATH = pathlib.Path("outputs/databases")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--remove_2022",
    "-r",
    action="store_true",
    default=True,
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

# for gid in selected_group_ids:
gid = selected_group_ids[0]

dfg = df_daily[df_daily["group_id"] == gid]

# how many guardians responded before this guardian
dfg["guardian_id_order"] = dfg["guardian_ids"].apply(len)
dfg["guardian_id_order"] = dfg["guardian_id_order"].astype(int).apply(range)
dfg = dfg.explode(["guardian_ids", "guardian_id_order"])

# if no one responds at all, assume order = 0
dfg["guardian_id_order"] = dfg["guardian_id_order"].fillna(0)

# how many messages were sent before guardian first message
# assume value 0 for all if no message was sent
dfg["cumulative_messages_before_response"] = dfg.apply(
    lambda x: x["guardian_ids_history"].index(x["guardian_ids"]) if pd.notnull(x["guardian_ids"]) else 0, axis=1
)

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

        dfg[col] += df_hour[3 * h + i]

        # fill nas with 0, i.e. all hour values are 0 for days without interventions
        dfg[col] = dfg[col].fillna(0)

    # map non zeroes to 1
    dfg[col] = dfg[col].astype(bool).astype(int)


# guardian level response regression: i.e. outcome is with respect to single guardians, not daily aggregate
X = []
y = []

features = ["n_guardians"]
features.extend(dfg.columns[5:])
# features.remove("Other")
features.remove("0-2")
features.remove("weekday")
features.remove("DA_intervention_hours")

guardian_ids = dfg["guardian_ids"].dropna().unique()

for day in sorted(dfg["sent_day"].dropna().unique()):

    dfday = dfg[dfg["sent_day"] == day]

    for guardian in guardian_ids:

        if guardian in dfday["guardian_ids"].dropna().unique():

            y.append(1)
            row = dfday[dfday["guardian_ids"] == guardian][features].values[0].tolist()

        else:

            y.append(0)
            row = [0] * len(features)

        row.extend([day, guardian])
        X.append(row)

features.extend(["date", "guardian_id"])

X = pd.DataFrame(X)
X.columns = features
y = np.array(y)

for col in X.columns[1:9]:
    X[col] = X[col].astype(int)

intervention_values = X[X.columns[1:9]].sum().values
top3_interventions = X.columns[1:9][intervention_values.argsort()[-3:][::-1]]
remove = [col for col in X.columns[1:9] if col not in top3_interventions]
X = X.drop(columns=remove)

# normalize some columns
normalize_cols = [
    "n_guardians",
    "n_moderator_messages",
    "n_distinct_moderators",
    "guardian_id_order",
    "cumulative_messages_before_response",
    "daily_total_messages",
]
# X = general_utils.normalize(X, normalize_cols)

X["constant"] = 1

# do a regression for each guardian

for guardian in dfg["guardian_ids"].dropna().unique():

    X_guardian = X[X["guardian_id"] == guardian]
    y_guardian = y[X["guardian_id"] == guardian]

    X_guardian = X_guardian.drop(columns=["date", "guardian_id"])

    X_guardian = general_utils.normalize(X_guardian, normalize_cols)

    # remove columns that sum up to 0
    X_guardian = X_guardian.loc[:, (X_guardian.sum(axis=0) != 0)]

    breakpoint()
    # TODO: do regression at guardian level
