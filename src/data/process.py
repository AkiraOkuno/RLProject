import argparse
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

warnings.simplefilter(action="ignore", category=FutureWarning)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv",
    action="store_true",
    help="Save processed data in csv format",
)
parser.add_argument(
    "--pickle",
    action="store_true",
    help="Save processed data in pickle format",
)
args = parser.parse_args()

RAW_PATH = pathlib.Path("data/raw")

OUTPUT_PATH = pathlib.Path("data/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# import data
print("Reading moderator data...")
with open(RAW_PATH / "df_moderator_22.json", "r") as f:
    data_moderator = json.load(f, strict=False)

df_moderator = pd.DataFrame(data_moderator)
del data_moderator

print("Reading guardian data...")
df_guardian = pd.read_json(RAW_PATH / "df_guardian_22.json")

print("Reading intervention data...")
df_interventions = pd.read_json(RAW_PATH / "df_interventions_22.json")

print("Preprocessing data...")
# remove rows with non numeric data in id columns
df_guardian = df_guardian[pd.to_numeric(df_guardian["interactions_id"], errors="coerce").notnull()]
df_moderator = df_moderator[pd.to_numeric(df_moderator["interactions_id"], errors="coerce").notnull()]
df_interventions = df_interventions[pd.to_numeric(df_interventions["interventions_id"], errors="coerce").notnull()]

# preprocess dataframes
df_guardian["message_type"] = df_guardian["message_type"].replace(
    {"chat": "Chat", "audio": "Audio", "document": "Document", "Voice": "Audio"}
)
df_moderator["message_type"] = df_moderator["message_type"].replace(
    {"chat": "Chat", "audio": "Audio", "document": "Document", "Voice": "Audio"}
)

# school level data
# can be matched to other data by group id
print("Reading school data and merging with interventions data...")

df_schools = pd.read_json(RAW_PATH / "df_groups_schools_22.json")
df_schools = df_schools.rename({"id": "groups_id"}, axis=1)
df_interventions = df_interventions.merge(df_schools, on="groups_id", how="left")

print("Concatenating data...")
df = pd.concat([df_guardian, df_moderator, df_interventions])
df["sent_time"] = pd.to_datetime(df["sent_time"])

# order stacked dataframe by timestamp
df = df.sort_values("sent_time")

# reset indexes
df = df.reset_index(drop=True)

print("Partial saving...")

if args.pickle:
    general_utils.save_pickle(df_interventions, OUTPUT_PATH / "df_interventions.pickle")
    general_utils.save_pickle(df_moderator, OUTPUT_PATH / "df_moderator.pickle")
    general_utils.save_pickle(df_guardian, OUTPUT_PATH / "df_guardian.pickle")
    general_utils.save_pickle(df, OUTPUT_PATH / "df_merge.pickle")

if args.csv:
    df_interventions.to_csv(OUTPUT_PATH / "df_interventions.csv", index=False)

    # following line bugs, fix:
    df_moderator.to_csv(OUTPUT_PATH / "df_moderator.csv", index=False)
    df_guardian.to_csv(OUTPUT_PATH / "df_guardian.csv", index=False)
    df.to_csv(OUTPUT_PATH / "df_merge.csv", index=False)

# remove saved data
del df_interventions, df_moderator, df_guardian, df

print("Process guardian_date_left dataframe and activities data...")

# process basic features from df_guardians_date_left
dfl = pd.read_json(RAW_PATH / "df_guardians_date_left_22.json")

dfl_processed = pd.DataFrame()

for gid in tqdm(dfl["groups_id"].dropna().unique()):

    dfg = dfl[dfl["groups_id"] == gid]
    dfg = dfg.sort_values(["date_joined"])
    dfg["current_size"] = range(1, dfg.shape[0] + 1)
    dfg["current_members"] = dfg["guardian_id"].map(lambda x: [x]).cumsum().apply(set)

    for row in dfg[["date_left", "guardian_id"]].dropna().iterrows():

        date = row[1]["date_left"]
        guardian = set([row[1]["guardian_id"]])

        dfg.loc[dfg["date_joined"] >= date, "current_size"] -= 1
        dfg.loc[dfg["date_joined"] >= date, "current_members"] -= guardian

        # add a new row for the dates where guardians left
        # pickup from last date before they left and update the information
        last_date_row = dfg.loc[dfg["date_joined"] <= date, :].values[-1].tolist()

        # update date
        last_date_row[2] = date

        # update current size
        last_date_row[5] -= 1

        # update current_members
        last_date_row[-1] -= guardian

        # append to dataframe
        dfg = dfg.append(dict(zip(dfg.columns, last_date_row)), ignore_index=True)

    dfl_processed = pd.concat([dfl_processed, dfg])

# finish preprocessing details
dfl_processed = dfl_processed.sort_values(["groups_id", "date_joined"])
dfl_processed = dfl_processed.drop(columns=["guardian_id", "date_left", "organization_id"])
dfl_processed = dfl_processed.rename({"date_joined": "date"}, axis=1)

print("Partial saving...")

# save to processed folder
if args.pickle:
    general_utils.save_pickle(dfl_processed, OUTPUT_PATH / "df_ins_and_outs.pickle")

if args.csv:
    dfl_processed.to_csv(OUTPUT_PATH / "df_ins_and_outs.csv", index=False)

del dfl_processed

# process activities data
# description of activies for each activity id, e.g. type, difficulty, audience
print("Reading activities data...")
df_activities = pd.read_json(RAW_PATH / "df_activities_22.json")

# map between intervention_id and activity_id
print("Reading interventions activity data...")
df_interventions_activity = pd.read_json(RAW_PATH / "df_intervention_activity_data_22.json")

# drop sub domain because for the same activity id, there are multiple sub domains
print("Processing data...")
df_activities = df_activities.drop(columns=["sub_domain"]).drop_duplicates()

# merge interventions data with activity information
# note: there are 8 existing activiy ids in df_interventions_activity without a counterpart in df_activities
# namely: 99, 230, 1478, 779, 14, 1713, 439, 25
df_activities = df_interventions_activity.merge(df_activities, on="activity_id", how="left")

# replace string "None" by nan
df_activities["learning_domain"] = df_activities["learning_domain"].replace("None", np.nan)
del df_interventions_activity

print("Final saving...")

# save to processed folder
if args.pickle:
    general_utils.save_pickle(df_activities, OUTPUT_PATH / "df_activities.pickle")

if args.csv:
    df_activities.to_csv(OUTPUT_PATH / "df_activities.csv", index=False)

print("Done!\n")
