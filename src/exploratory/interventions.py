import argparse
import pathlib
import random
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--random_groups",
    "-rg",
    default=5,
    help="Number of random group ids to do analysis",
    type=int,
)
args = parser.parse_args()


DATA_PATH = pathlib.Path("data/processed")

PLOTS_PATH = pathlib.Path("outputs/plots/DA_interventions")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

STATS_PATH = pathlib.Path("outputs/statistics/DA_interventions")
STATS_PATH.mkdir(parents=True, exist_ok=True)


def n_distant_interventions(x: pd.Series):

    """
    Function that takes a pd.Series input of timestamps and returns the number of DA interventions with a difference of more than 60s with respect to previous intervention.
    """

    if x.shape[0] == 0:
        n_distant_interventions = 0
    elif x.shape[0] == 1:
        n_distant_interventions = 1
    else:
        diff = (x - x.shift())[1:]
        diff_in_secs = [t.seconds for t in diff]
        n_distant_interventions = np.sum(np.array(diff_in_secs) > 60)
        n_distant_interventions += 1

    return n_distant_interventions


# import data
df_interventions = pd.read_csv(DATA_PATH / "df_interventions.csv")

df_interventions["sent_time"] = pd.to_datetime(df_interventions["sent_time"])
df_interventions["sent_day"] = df_interventions["sent_time"].dt.strftime("%Y-%m-%d")

df_interventions = df_interventions.sort_values(["groups_id", "sent_time"])

dfda = df_interventions[df_interventions["intervention_type"] == "DA"]

dfda = (
    dfda.groupby(["groups_id", "sent_day"])["sent_time"].aggregate(lambda x: n_distant_interventions(x)).reset_index()
)
dfda["sent_day"] = pd.to_datetime(dfda["sent_day"])
dfda.columns = ["groups_id", "sent_day", "DA_count"]


for gid in random.sample(list(dfda["groups_id"].dropna().unique()), args.random_groups):

    dfg = dfda[dfda["groups_id"] == gid]

    plt.figure(figsize=(12, 5))
    plt.plot(dfg["sent_day"], dfg["DA_count"])
    plt.plot(dfg["sent_day"], dfg["DA_count"], "go", markersize=3)
    plt.title(f"Group {gid} - Number of DA interventions per day")
    print(PLOTS_PATH / f"group_{gid}-DA_interventions_per_day.png")
    plt.savefig(PLOTS_PATH / f"group_{gid}-DA_interventions_per_day.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    da_count = dfg["DA_count"].value_counts().sort_index(ascending=True)
    plt.bar(da_count.index, da_count.values)
    plt.xticks(da_count.index)
    plt.title(f"Group {gid} - Barplot of number of DA interventions per day")
    print(PLOTS_PATH / f"group_{gid}-DA_interventions_per_day-barplot.png")
    plt.savefig(PLOTS_PATH / f"group_{gid}-DA_interventions_per_day-barplot.png")
    plt.close()

    time.sleep(0.1)
