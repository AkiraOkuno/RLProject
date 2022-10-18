import argparse
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

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

print("Concatenating data...")
df = pd.concat([df_guardian, df_moderator, df_interventions])
df["sent_time"] = pd.to_datetime(df["sent_time"])

# order stacked dataframe by timestamp
df = df.sort_values("sent_time")

# reset indexes
df = df.reset_index(drop=True)

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

print("Saving data...")

# save to processed folder
if args.pickle:
    general_utils.save_pickle(df_interventions, OUTPUT_PATH / "df_interventions.pickle")
    general_utils.save_pickle(df_moderator, OUTPUT_PATH / "df_moderator.pickle")
    general_utils.save_pickle(df_guardian, OUTPUT_PATH / "df_guardian.pickle")
    general_utils.save_pickle(df, OUTPUT_PATH / "df_merge.pickle")
    general_utils.save_pickle(dfl_processed, OUTPUT_PATH / "df_ins_and_outs.pickle")

if args.csv:
    df_interventions.to_csv(OUTPUT_PATH / "df_interventions.csv", index=False)
    df_moderator.to_csv(OUTPUT_PATH / "df_moderator.csv", index=False)
    df_guardian.to_csv(OUTPUT_PATH / "df_guardian.csv", index=False)
    df.to_csv(OUTPUT_PATH / "df_merge.csv", index=False)
    dfl_processed.to_csv(OUTPUT_PATH / "df_ins_and_outs.csv", index=False)

print("Done!\n")
