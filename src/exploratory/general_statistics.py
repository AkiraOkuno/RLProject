import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


DATA_PATH = pathlib.Path("data/processed")
OUTPUT_PATH = pathlib.Path("outputs/statistics")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# import data
df_guardian = pd.read_csv(DATA_PATH / "df_guardian.csv")
df_moderator = pd.read_csv(DATA_PATH / "df_moderator.csv")
df_interventions = pd.read_csv(DATA_PATH / "df_interventions.csv")
df = pd.read_csv(DATA_PATH / "df_merge.csv")

# exploratory analysis
df_guardian["message_type"].value_counts()
df_guardian["message_type"].value_counts() / df_guardian.shape[0]
df_guardian[df_guardian["message_type"] == "Audio"].describe()
df_guardian[df_guardian["message_type"] == "Audio"]["duration"].hist(bins=50)


df_moderator["message_type"].value_counts()
df_moderator["message_type"].value_counts() / df_moderator.shape[0]
df_moderator[df_moderator["message_type"] == "Audio"].describe()
df_moderator[df_moderator["message_type"] == "Audio"]["duration"].hist(bins=50)

# n guardians per group
plt.hist(df.groupby("groups_id")["guardian_id"].nunique(), bins=50)
plt.title("Number of guardians per group")
plt.show()
