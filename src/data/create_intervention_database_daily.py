import argparse
import os
import pathlib
import sys
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

DATA_PATH = pathlib.Path("data/processed")
OUTPUT_PATH = pathlib.Path("outputs/databases")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH / "df_merge.csv")

df["sent_time"] = pd.to_datetime(df["sent_time"])
df["guardian_id"] = df["guardian_id"].astype("Int64")

df["action"] = None
df.loc[~np.isnan(df["interventions_id"].values), "action"] = "intervention"
df.loc[~np.isnan(df["guardian_id"].values), "action"] = "guardian"
df.loc[~np.isnan(df["moderator_id"].values), "action"] = "moderator"

df["current_intervention_type"] = df["intervention_type"].fillna(method="ffill")
df["sent_day"] = pd.to_datetime(df["sent_time"].dt.strftime("%Y-%m-%d"))

intervention_types = df["intervention_type"].dropna().unique()

X = []

for gid in tqdm(df["groups_id"].dropna().unique()):

    # filter only specific group
    dfg = df[df["groups_id"] == gid]
    df_guardian = dfg[dfg["action"] == "guardian"]

    for day in sorted(dfg["sent_day"].unique()):

        output = []
        output.append(gid)
        output.append(day)

        dfday = dfg[dfg["sent_day"] == day]

        n_guardians = dfday["guardian_id"].dropna().nunique()
        output.append(n_guardians)

        # ensure to save guardian ids in order of response
        guardian_ids = dfday.sort_values("sent_time")["guardian_id"].dropna().unique().tolist()
        guardian_ids_full_history = dfday.sort_values("sent_time")["guardian_id"].dropna().tolist()
        output.append(guardian_ids)
        output.append(guardian_ids_full_history)

        day_interventions = set(dfday["intervention_type"].dropna().unique())

        for type in intervention_types:
            output.append(type in day_interventions)

        da_intervention_hours = dfday[dfday["intervention_type"] == "DA"]["sent_time"].dt.hour.unique().tolist()
        output.append(da_intervention_hours)

        n_moderator_messages = dfday[dfday["action"] == "moderator"].shape[0]
        n_distinct_moderators = len(dfday["moderator_id"].dropna().unique())
        output.extend([n_moderator_messages, n_distinct_moderators])

        X.append(output)

X = pd.DataFrame(X)
colnames = ["group_id", "sent_day", "n_guardians", "guardian_ids", "guardian_ids_history"]
colnames.extend(intervention_types)
colnames.extend(["DA_intervention_hours", "n_moderator_messages", "n_distinct_moderators"])
X.columns = colnames

df_hour_dummies = pd.get_dummies(X["DA_intervention_hours"].apply(pd.Series).stack()).sum(level=0)
X = pd.concat([X, df_hour_dummies], axis=1).fillna(0)

X["weekday"] = X["sent_day"].dt.weekday

general_utils.save_pickle(X, OUTPUT_PATH / "daily_features_database.pickle")

breakpoint()

# simple stats -> NEED TO GO TO ANOTHER SCRIPT
X[X["DA"] is True]["n_guardians"].mean()

X[X["DA"] is False]["n_guardians"].mean()

X.groupby("weekday")["n_guardians"].mean()

X["n_moderator_messages_mod10"] = X["n_moderator_messages"] - X["n_moderator_messages"] % 10 + 10
X.groupby("n_moderator_messages_mod10")["n_guardians"].mean()

for type in intervention_types:
    X.groupby(type)["n_guardians"].mean()
