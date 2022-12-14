import argparse
import os
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--group_id",
    "-g",
    default=29782,
    help="Group id to do analysis",
    type=int,
)
group.add_argument(
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


def n_distant_interventions(x: pd.Series):

    """
    Function that takes a pd.Series input of timestamps and returns the number of DA interventions with a difference of more than 60s with respect to previous intervention.
    """

    if x.shape[0] == 1:
        n_distant_interventions = 1
    else:
        diff = (x - x.shift())[1:]
        diff_in_secs = [t.seconds for t in diff]
        n_distant_interventions = np.sum(np.array(diff_in_secs) > 60)
        n_distant_interventions += 1

    return n_distant_interventions


# import data
df_interventions = general_utils.open_pickle(DATA_PATH / "df_interventions.pickle")
df_guardian = general_utils.open_pickle(DATA_PATH / "df_guardian.pickle")

# preprocessing
df_guardian["sent_time"] = pd.to_datetime(df_guardian["sent_time"])
df_guardian["sent_day"] = df_guardian["sent_time"].dt.strftime("%Y-%m-%d")

df_interventions["sent_time"] = pd.to_datetime(df_interventions["sent_time"])
df_interventions["sent_day"] = df_interventions["sent_time"].dt.strftime("%Y-%m-%d")
df_interventions["sent_hr"] = df_interventions["sent_time"].dt.hour.astype(str)
df_interventions["sent_date"] = pd.to_datetime(df_interventions["sent_time"].dt.date)

df_guardian = df_guardian.sort_values(["groups_id", "sent_time"])
df_interventions = df_interventions.sort_values(["groups_id", "sent_time"])

# work with DA restricted interventions
dfda = df_interventions[df_interventions["intervention_type"] == "DA"]

dfda = (
    dfda.groupby(["groups_id", "sent_day"])["sent_time"].aggregate(lambda x: n_distant_interventions(x)).reset_index()
)
dfda["sent_day"] = pd.to_datetime(dfda["sent_day"])
dfda.columns = ["groups_id", "sent_day", "DA_count"]

# fill missing dates in dfda, i.e. dates where there where no DA interventions with 0
new_dfda = pd.DataFrame()

for gid in dfda["groups_id"].dropna().unique():

    dfg = dfda[dfda["groups_id"] == gid]
    min_date = dfg["sent_day"].min()
    max_date = dfg["sent_day"].max()
    idx = pd.date_range(min_date, max_date)
    da_count_series = dfg["DA_count"]
    da_count_series.index = pd.DatetimeIndex(dfg["sent_day"])
    da_count_series = da_count_series.reindex(idx, fill_value=0)

    dfg = pd.DataFrame(da_count_series)
    dfg["groups_id"] = gid
    dfg = dfg.reset_index().rename(columns={"index": "sent_day"})

    new_dfda = pd.concat([new_dfda, dfg])

dfda = new_dfda.copy()
del new_dfda

# check for number of responses in days with different DA levels
df_guardian_responses = df_guardian.groupby(["groups_id", "sent_day"])["guardian_id"].nunique().reset_index()
df_guardian_responses = df_guardian_responses.rename(columns={"guardian_id": "n_guardian_responses"})

dfda["sent_day"] = dfda["sent_day"].astype(str)
df_guardian_responses["sent_day"] = df_guardian_responses["sent_day"].astype(str)

df_guardian_responses = pd.merge(dfda, df_guardian_responses, how="left")
df_guardian_responses["n_guardian_responses"] = df_guardian_responses["n_guardian_responses"].fillna(0)

df_guardian_responses["n_guardian_responses"] = df_guardian_responses["n_guardian_responses"].astype(int)

# calculate mean responses in each group for each number of daily DA interventions
df_guardian_responses = (
    df_guardian_responses.groupby(["groups_id", "DA_count"])["n_guardian_responses"].mean().reset_index()
)

# calculate mean response across groups for each number of daily DA interventions
print(df_guardian_responses.groupby("DA_count")["n_guardian_responses"].mean())

######################################################################################################################

# group temporal analysis with polar plots

df_guardian["sent_day"] = df_guardian["sent_time"].dt.day
df_guardian["sent_dayofweek"] = df_guardian["sent_time"].dt.dayofweek
df_guardian["sent_date"] = pd.to_datetime(df_guardian["sent_time"].dt.date)
df_guardian["sent_hr"] = df_guardian["sent_time"].dt.hour
df_guardian["sent_month"] = df_guardian["sent_time"].dt.month
df_guardian["sent_week"] = df_guardian["sent_time"].dt.week

if args.random_groups:
    selected_group_ids = random.sample(list(df_guardian["groups_id"].dropna().unique()), args.random_groups)
elif args.group_id:
    selected_group_ids = [args.group_id]
else:
    raise ValueError("Group choice method not implemented yet")

for gid in tqdm(selected_group_ids):

    dfg = df_guardian[df_guardian["groups_id"] == gid]

    # how many unique guardians respond in a day at at given hour on average:
    df_hour = (
        dfg.groupby(["sent_date", "sent_hr"])["guardian_id"]
        .nunique()
        .reset_index()
        .groupby("sent_hr")["guardian_id"]
        .mean()
        .reset_index()
    )
    df_hour["sent_hr"] = df_hour["sent_hr"].astype(str)

    dfg_interventions = df_interventions[df_interventions["groups_id"] == gid]
    dfg_da = dfg_interventions[dfg_interventions["intervention_type"] == "DA"]
    hour_distribution = (
        dfg_da.groupby(["sent_date", "sent_hr"]).size().reset_index()["sent_hr"].value_counts(normalize=True)
    )
    hour_distribution = hour_distribution / hour_distribution.max()
    hour_distribution = hour_distribution.reset_index()

    hour_distribution.columns = ["sent_hr", "weight"]

    # complete possibly missing hours with zero in hour_distribution
    hour_distribution = df_hour.merge(hour_distribution, on="sent_hr", how="left").fillna(0)[["sent_hr", "weight"]]

    # rename weight to guardian id just to match column names with guardian data
    hour_distribution.columns = ["sent_hr", "guardian_id"]
    hour_distribution["type"] = "DA intervention"

    df_hour["type"] = "guardian"

    df_concat = pd.concat([df_hour, hour_distribution])

    fig = px.line_polar(df_concat, r="guardian_id", theta="sent_hr", color="type", line_close=True)
    fig.update_polars(angularaxis_type="category")
    fig.update_traces(fill="toself")
    fig.update_layout(title_text=f"Group {gid} - Hourly guardian response average - n={dfg.shape[0]}", title_x=0.5)

    print(PLOTS_PATH / f"group_{gid}-hourly_polar_plot.png")
    fig.write_image(PLOTS_PATH / f"group_{gid}-hourly_polar_plot.png")

    df_hour = df_hour.drop(columns=["type"])

    df_month = (
        dfg.groupby(["sent_date", "sent_month"])["guardian_id"]
        .nunique()
        .reset_index()
        .groupby("sent_month")["guardian_id"]
        .mean()
        .reset_index()
    )
    df_month["sent_month"] = df_month["sent_month"].astype(str)

    fig = px.line_polar(df_month, r="guardian_id", theta="sent_month", line_close=True)
    fig.update_polars(angularaxis_type="category")
    fig.update_traces(fill="toself")
    fig.update_layout(title_text=f"Group {gid} - Monthly guardian response average - n={dfg.shape[0]}", title_x=0.5)

    print(PLOTS_PATH / f"group_{gid}-monthly_polar_plot.png")
    fig.write_image(PLOTS_PATH / f"group_{gid}-monthly_polar_plot.png")

    df_weekday = (
        dfg.groupby(["sent_date", "sent_dayofweek"])["guardian_id"]
        .nunique()
        .reset_index()
        .groupby("sent_dayofweek")["guardian_id"]
        .mean()
        .reset_index()
    )
    df_weekday["sent_dayofweek"] = df_weekday["sent_dayofweek"].astype(str)

    fig = px.line_polar(df_weekday, r="guardian_id", theta="sent_dayofweek", line_close=True)
    fig.update_polars(angularaxis_type="category")
    fig.update_traces(fill="toself")
    fig.update_layout(title_text=f"Group {gid} - Weekday guardian response average - n={dfg.shape[0]}", title_x=0.5)

    print(PLOTS_PATH / f"group_{gid}-weekday_polar_plot.png")
    fig.write_image(PLOTS_PATH / f"group_{gid}-weekday_polar_plot.png")

    ######################################################################################################################

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
