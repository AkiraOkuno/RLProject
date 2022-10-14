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

# how many messages were sent before guardian first message
# assume value 0 for all if no message was sent
dfg["cumulative_messages_before_response"] = dfg.apply(
    lambda x: x["guardian_ids_history"].index(x["guardian_ids"]) if pd.notnull(x["guardian_ids"]) else 0, axis=1
)

# daily feature: number of messages in that day
dfg["daily_total_messages"] = dfg["guardian_ids_history"].apply(len)

# guardian level response regression: i.e. outcome is with respect to single guardians, not daily aggregate
X = []
y = []

features = ["n_guardians"]
features.extend(dfg.columns[5:])
features.remove("Other")
features.remove(0.0)
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

# TODO: do regression
