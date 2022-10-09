import argparse
import pathlib
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

# filter only specific group
df = df[df["groups_id"] == args.group_id]
df_guardian = df[df["action"] == "guardian"]

INITIAL_DAYS = 30
init_date = df["sent_time"].min() + timedelta(days=INITIAL_DAYS)
intervention_dates = df[(df["action"] == "intervention") & (df["sent_time"] > init_date)]["sent_time"].values

X = []

current_guardians = set()
cumulative_responses = 0
previous_guardian = 0
previous_moderator = 0
last_intervention_time = None
last_intervention_type = "0"
last_action = None
all_guardians = set(df["guardian_id"].dropna().unique())

for _, row in df[df["sent_time"] > init_date].iterrows():

    if row["action"] == "guardian" and row["guardian_id"] not in current_guardians:

        output = []

        output.append(1)
        output.append(row["guardian_id"])
        output.append(row["current_intervention_type"])
        output.append(last_intervention_time)
        output.append(cumulative_responses)
        output.append(previous_guardian)
        output.append(previous_moderator)
        output.append(row["sent_time"])

        X.append(output)

        # TODO: add if some specific nodes already responded

        current_guardians.add(row["guardian_id"])
        cumulative_responses += 1
        previous_guardian = row["guardian_id"]

        last_action = "guardian"

    elif row["action"] == "intervention" and last_action != "intervention":

        for guardian in all_guardians - current_guardians:

            output = []
            output.append(0)
            output.append(guardian)
            output.append(last_intervention_type)
            output.append(last_intervention_time)
            output.append(cumulative_responses)
            output.append(previous_guardian)
            output.append(previous_moderator)
            output.append(row["sent_time"])

            X.append(output)

        current_guardians = set()
        previous_guardian = 0
        last_intervention_time = row["sent_time"].hour
        cumulative_responses = 0
        last_action = "intervention"
        last_intervention_type = row["intervention_type"]

    elif row["action"] == "moderator":

        previous_moderator = row["moderator_id"]
        last_action = "moderator"

    else:
        pass

X = pd.DataFrame(X)
X.columns = [
    "response",
    "guardian_id",
    "intervention_type",
    "intervention_hour",
    "n_previous_guardians",
    "last_guardian_id",
    "last_moderator_id",
    "time_of_response",
]

# filter data only after first intervention is observed
X = X[~X["intervention_hour"].isna()].reset_index(drop=True)

X["last_guardian_id"] = X["last_guardian_id"].astype("Int64")
X["last_moderator_id"] = X["last_moderator_id"].astype("Int64")

# discretize intervention hour into 3 hour intervals # 0 = {0,1,2}, 3={3,4,5}, ..., 21={21,22,23}
X["intervention_hour"] = X["intervention_hour"].apply(lambda x: x - x % 3)

# add dummies
X = pd.concat([X, pd.get_dummies(X["intervention_type"], prefix="I", drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X["intervention_hour"], prefix="hour", drop_first=True)], axis=1)
# X = pd.concat([X, pd.get_dummies(X["last_guardian_id"], prefix = "gid", drop_first=True)], axis=1)
# X = pd.concat([X, pd.get_dummies(X["last_moderator_id"], prefix = "last-mid", drop_first=True)], axis=1)

y = X["response"]
X = X.drop(columns=["response", "intervention_hour", "last_guardian_id", "last_moderator_id", "time_of_response"])

# full regression
top3_interventions = X["intervention_type"].value_counts().index[:3].values
remove = [col for col in X.columns if col[0] == "I" and col[2:] not in top3_interventions]
Xf = X.drop(columns=remove)
Xf = Xf.drop("intervention_type", axis=1)
Xf = Xf.drop("guardian_id", axis=1)
Xf["constant"] = 1

reg = sm.Logit(y, Xf).fit()

breakpoint()

for gid in X["guardian_id"].unique():

    # remove guardian id from features
    Xg = X[X["guardian_id"] == gid].iloc[:, 1:]

    yg = y[X["guardian_id"] == gid]

    top3_interventions = Xg["intervention_type"].value_counts().index[:3].values
    remove = [col for col in Xg.columns if col[0] == "I" and col[2:] not in top3_interventions]
    Xg = Xg.drop(columns=remove)
    Xg = Xg.drop("intervention_type", axis=1)
    Xg["constant"] = 1

    reg = sm.Logit(yg, Xg).fit()

    breakpoint()
