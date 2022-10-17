import argparse
import json
import os
import pathlib
import sys

import pandas as pd

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

print("Saving data...")

# save to processed folder
if args.pickle:
    general_utils.save_pickle(df_interventions, OUTPUT_PATH / "df_interventions.pickle")
    general_utils.save_pickle(df_moderator, OUTPUT_PATH / "df_moderator.pickle")
    general_utils.save_pickle(df_guardian, OUTPUT_PATH / "df_guardian.pickle")
    general_utils.save_pickle(df, OUTPUT_PATH / "df_merge.pickle")

if args.csv:
    df_interventions.to_csv(OUTPUT_PATH / "df_interventions.csv", index=False)
    df_moderator.to_csv(OUTPUT_PATH / "df_moderator.csv", index=False)
    df_guardian.to_csv(OUTPUT_PATH / "df_guardian.csv", index=False)
    df.to_csv(OUTPUT_PATH / "df_merge.csv", index=False)

print("Done!\n")
