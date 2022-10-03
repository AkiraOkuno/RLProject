from src.utils import general_utils

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import argparse
import pathlib
import random
from datetime import datetime
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--initial_month", "-mi", default=1, help="Month to do analysis", type=int,
)
parser.add_argument(
    "--initial_year", "-yi", default=2022, help="Year to do analysis", type=int,
)
parser.add_argument(
    "--final_month", "-mf", default=2, help="Month to do analysis", type=int,
)
parser.add_argument(
    "--final_year", "-yf", default=2022, help="Year to do analysis", type=int,
)
parser.add_argument(
    "--colored_interventions",
    "-ci",
    help="whether to plot different interventions with different colors",
    action="store_true",
)
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--group_id", "-g", default=29782, help="Group id to do analysis", type=int,
)
group.add_argument(
    "--random_groups",
    "-rg",
    help="Number of random group ids to do analysis",
    type=int,
)
group.add_argument(
    "--all_groups", "-ag", help="All group ids are analyzed", action="store_true",
)
args = parser.parse_args()

DATA_PATH = pathlib.Path("data/processed")

df = pd.read_csv(DATA_PATH / "df_merge.csv")

df["sent_time"] = pd.to_datetime(df["sent_time"])

df["action"] = None
df.loc[~np.isnan(df["interventions_id"].values), "action"] = "intervention"
df.loc[~np.isnan(df["guardian_id"].values), "action"] = "guardian"
df.loc[~np.isnan(df["moderator_id"].values), "action"] = "moderator"

# filter dates
df = df[
    (df["sent_time"] > datetime(args.initial_year, args.initial_month, 1))
    & (
        df["sent_time"]
        < general_utils.last_day_of_month(
            datetime(args.final_year, args.final_month, 1)
        )
    )
]


def group_plot(group_id, mi, yi, mf, yf, ci, data=df):

    # filter group id
    df_group = data[data["groups_id"] == group_id]

    # x axis = dates
    dates = df_group["sent_time"]

    plt.style.use("ggplot")

    ########################################################################################################

    # calculate entry of new guardians
    cumulative_new_guardians = general_utils.cumulative_distinct_values(
        df_group["guardian_id"]
    )
    breakpoint()
    plt.figure(figsize=(22, 7))
    plt.plot(dates, cumulative_new_guardians.values, color="r")

    if ci:
        # plot same interventions with same colors

        interventions = df_group["interventions_id"].unique()
        color = cm.rainbow(np.linspace(0, 1, len(interventions)))
        intervention_colors_dict = dict(zip(interventions, color))
        intervention_df = df_group[df_group["action"] == "intervention"]
        ymax = plt.gca().get_ylim()[1]

        for iid in intervention_df["interventions_id"].dropna().unique():

            timestamps = intervention_df[intervention_df["interventions_id"] == iid][
                "sent_time"
            ].values

            plt.vlines(
                x=timestamps,
                ymin=0,
                ymax=ymax,
                color=intervention_colors_dict[iid],
                ls="--",
                lw=0.7,
                alpha=0.5,
            )
    else:

        plt.vlines(
            x=df_group[df_group["action"] == "intervention"]["sent_time"].values,
            ymin=0,
            ymax=plt.gca().get_ylim()[1],
            color="grey",
            ls="--",
            lw=0.7,
            alpha=0.5,
        )

    plt.title(f"Group {int(group_id)} - Cumulative new guardians")
    plt.savefig(
        f"outputs/plots/group_{group_id}-mi_{mi}-yi_{yi}-mf_{mf}-yf_{yf}-ci_{ci}-cumulative_new_guardians.png"
    )

    ########################################################################################################

    # plot cumulative number of messages of guardians and moderators
    cumulative_guardian_messages = np.cumsum(df_group["action"] == "guardian").values
    cumulative_moderator_messages = np.cumsum(df_group["action"] == "moderator").values

    plt.figure(figsize=(22, 7))
    plt.plot(dates, cumulative_guardian_messages, color="r", label="guardian")
    plt.plot(dates, cumulative_moderator_messages, color="g", label="moderator")
    plt.title(
        f"Group {int(group_id)} - Cumulative messages from guardians and moderators"
    )
    plt.legend(bbox_to_anchor=(1.125, 1.15), loc="upper right")

    if ci:
        # plot same interventions with same colors

        interventions = df_group["interventions_id"].unique()
        color = cm.rainbow(np.linspace(0, 1, len(interventions)))
        intervention_colors_dict = dict(zip(interventions, color))
        intervention_df = df_group[df_group["action"] == "intervention"]
        ymax = plt.gca().get_ylim()[1]

        for iid in intervention_df["interventions_id"].dropna().unique():

            timestamps = intervention_df[intervention_df["interventions_id"] == iid][
                "sent_time"
            ].values

            plt.vlines(
                x=timestamps,
                ymin=0,
                ymax=ymax,
                color=intervention_colors_dict[iid],
                ls="--",
                lw=0.7,
                alpha=0.5,
            )
    else:

        plt.vlines(
            x=df_group[df_group["action"] == "intervention"]["sent_time"].values,
            ymin=0,
            ymax=plt.gca().get_ylim()[1],
            color="grey",
            ls="--",
            lw=0.7,
            alpha=0.5,
        )

    plt.savefig(
        f"outputs/plots/group_{group_id}-mi_{mi}-yi_{yi}-mf_{mf}-yf_{yf}-ci_{ci}-cumulative_messages.png"
    )

    ########################################################################################################

    # plot guardians cumulative messages individually
    guardian_ids = df_group["guardian_id"].dropna().unique()
    plt.figure(figsize=(22, 7))

    color = cm.rainbow(np.linspace(0, 1, len(guardian_ids)))

    for i, c in zip(range(len(guardian_ids)), color):
        cumulative_guardian_i_messages = np.cumsum(
            df_group["guardian_id"] == guardian_ids[i]
        ).values
        plt.plot(
            dates, cumulative_guardian_i_messages, color=c, label=int(guardian_ids[i])
        )

    plt.title(
        f"Group {int(group_id)} - Cumulative messages from all individual guardians"
    )
    plt.legend(bbox_to_anchor=(1.125, 1.15), loc="upper right")

    if ci:
        # plot same interventions with same colors

        interventions = df_group["interventions_id"].unique()
        color = cm.rainbow(np.linspace(0, 1, len(interventions)))
        intervention_colors_dict = dict(zip(interventions, color))
        intervention_df = df_group[df_group["action"] == "intervention"]
        ymax = plt.gca().get_ylim()[1]

        for iid in intervention_df["interventions_id"].dropna().unique():

            timestamps = intervention_df[intervention_df["interventions_id"] == iid][
                "sent_time"
            ].values

            plt.vlines(
                x=timestamps,
                ymin=0,
                ymax=ymax,
                color=intervention_colors_dict[iid],
                ls="--",
                lw=0.7,
                alpha=0.5,
            )
    else:

        plt.vlines(
            x=df_group[df_group["action"] == "intervention"]["sent_time"].values,
            ymin=0,
            ymax=plt.gca().get_ylim()[1],
            color="grey",
            ls="--",
            lw=0.7,
            alpha=0.5,
        )

    plt.savefig(
        f"outputs/plots/group_{group_id}-mi_{mi}-yi_{yi}-mf_{mf}-yf_{yf}-ci_{ci}-cumulative_individual_messages.png"
    )

    ########################################################################################################


if args.group_id:
    group_plot(
        args.group_id,
        args.initial_month,
        args.initial_year,
        args.final_month,
        args.colored_interventions,
        args.final_year,
    )

# plot random groups if flag random_groups is active
if args.random_groups:
    for gid in random.sample(list(df["groups_id"].unique()), args.random_groups):
        group_plot(
            gid,
            args.initial_month,
            args.initial_year,
            args.final_month,
            args.colored_interventions,
            args.final_year,
        )

if args.all_groups:
    for gid in tqdm(list(df["groups_id"].unique())):
        group_plot(
            gid,
            args.initial_month,
            args.initial_year,
            args.final_month,
            args.colored_interventions,
            args.final_year,
        )
