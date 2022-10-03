import pandas as pd

# import data
df_guardian = pd.read_csv("data/raw/df_guardian_22.csv")
df_moderator = pd.read_csv("data/raw/df_moderator_22.csv")
df_interventions = pd.read_csv("data/raw/interventions_22.csv")

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
df_interventions.to_csv("data/processed/df_interventions.csv")
df_moderator.to_csv("data/processed/df_moderator.csv")
df_guardian.to_csv("data/processed/df_guardian.csv")
df.to_csv("data/processed/df_merge.csv")
