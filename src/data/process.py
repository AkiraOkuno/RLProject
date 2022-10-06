import pathlib

import pandas as pd

RAW_PATH = pathlib.Path("data/raw")

OUTPUT_PATH = pathlib.Path("data/processed")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# import data
df_guardian = pd.read_csv(RAW_PATH / "df_guardian_22.csv")
df_moderator = pd.read_csv(RAW_PATH / "df_moderator_22.csv")

# use new data with intervention types
df_interventions = pd.read_csv("data/raw/df_interventions.csv")

# preprocess dataframes
df_guardian["message_type"] = df_guardian["message_type"].replace(
    {"chat": "Chat", "audio": "Audio", "document": "Document", "Voice": "Audio"}
)
df_moderator["message_type"] = df_moderator["message_type"].replace(
    {"chat": "Chat", "audio": "Audio", "document": "Document", "Voice": "Audio"}
)

df = pd.concat([df_guardian, df_moderator, df_interventions])
df["sent_time"] = pd.to_datetime(df["sent_time"])

# order stacked dataframe by timestamp
df = df.sort_values("sent_time")

# reset indexes
df = df.reset_index(drop=True)

# save to processed folder
df_interventions.to_csv(OUTPUT_PATH / "df_interventions.csv")
df_moderator.to_csv(OUTPUT_PATH / "df_moderator.csv")
df_guardian.to_csv(OUTPUT_PATH / "df_guardian.csv")
df.to_csv(OUTPUT_PATH / "df_merge.csv")
