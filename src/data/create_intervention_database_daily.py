import argparse
import pathlib
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--group_id",
    "-g",
    default=17824,
    help="Group id to do analysis",
    type=int,
)
args = parser.parse_args()

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

# filter only specific group
dfg = df[df["groups_id"] == args.group_id]
df_guardian = dfg[dfg["action"] == "guardian"]

X = []

for day in tqdm(sorted(dfg["sent_day"].unique())):

    output = []
    output.append(day)

    dfday = dfg[dfg["sent_day"] == day]

    n_guardians = dfday["guardian_id"].dropna().nunique()
    output.append(n_guardians)

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
colnames = ["sent_day", "n_guardians"]
colnames.extend(intervention_types)
colnames.extend(["DA_intervention_hours", "n_moderator_messages", "n_distinct_moderators"])
X.columns = colnames

df_hour_dummies = pd.get_dummies(X["DA_intervention_hours"].apply(pd.Series).stack()).sum(level=0)
X = pd.concat([X, df_hour_dummies], axis=1).fillna(0)

X["weekday"] = X["sent_day"].dt.weekday

# simple stats
X[X["DA"] is True]["n_guardians"].mean()

X[X["DA"] is False]["n_guardians"].mean()

X.groupby("weekday")["n_guardians"].mean()

X["n_moderator_messages_mod10"] = X["n_moderator_messages"] - X["n_moderator_messages"] % 10 + 10
X.groupby("n_moderator_messages_mod10")["n_guardians"].mean()

for type in intervention_types:
    X.groupby(type)["n_guardians"].mean()

breakpoint()
