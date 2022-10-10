import pathlib
from random import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--random_groups",
    "-rg",
    help="Number of random group ids to do analysis",
    type=int,
)
args = parser.parse_args()


DATA_PATH = pathlib.Path("data/processed")

PLOTS_PATH = pathlib.Path("outputs/plots/DA_interventions")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

STATS_PATH = pathlib.Path("outputs/statistics/DA_interventions")
STATS_PATH.mkdir(parents=True, exist_ok=True)

# import data
df_interventions = pd.read_csv(DATA_PATH / "df_interventions.csv")

df_interventions["sent_time"] = pd.to_datetime(df_interventions["sent_time"])
df_interventions["sent_day"] = df_interventions["sent_time"].dt.strftime("%Y-%m-%d")

# checks for each day and for each group the number of DA interventions
df_DA = df_interventions.groupby(["groups_id","sent_day"])["intervention_type"].aggregate(lambda x:sum(x == "DA")).reset_index()
df_DA["sent_day"] = pd.to_datetime(df_DA["sent_day"])
df_DA.columns = ["groups_id","sent_day","DA_count"]

for gid in random.sample(df_DA["groups_id"].dropna().unique(), args.random_groups):

    dfg = df_DA[df_DA["groups_id"]==gid]
    plt.plot(dfg["sent_day"], dfg["DA_count"])
    plt.savefig(PLOTS_PATH / f"group_{gid}-DA_interventions_per_day.png")