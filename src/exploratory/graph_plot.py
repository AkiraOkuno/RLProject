import argparse
import pathlib
import random

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

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
parser.add_argument(
    "--intervention_type",
    "-it",
    action="store_true",
    help="create one graph per intervention",
)
parser.add_argument(
    "--normalized_weights",
    "-nw",
    action="store_true",
    help="normalize weights to between 0 and 10",
)
args = parser.parse_args()

DATA_PATH = pathlib.Path("data/processed")
OUTPUT_PATH = pathlib.Path("outputs/plots/graphs")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_PATH / "df_merge.csv")

df["sent_time"] = pd.to_datetime(df["sent_time"])
df["guardian_id"] = df["guardian_id"].astype("Int64")

df["action"] = None
df.loc[~np.isnan(df["interventions_id"].values), "action"] = "intervention"
df.loc[~np.isnan(df["guardian_id"].values), "action"] = "guardian"
df.loc[~np.isnan(df["moderator_id"].values), "action"] = "moderator"

df["current_intervention_type"] = df["intervention_type"].fillna(method="ffill")

df_guardian = df[df["action"] == "guardian"]

group_ids = df_guardian["groups_id"].dropna().unique()
selected_group_ids = random.sample(list(group_ids), args.random_groups)

for gid in selected_group_ids:

    df_group = df_guardian[df_guardian["groups_id"] == gid]
    df_group = df_group[~df_group["current_intervention_type"].isna()]
    guardians = df_group["guardian_id"].dropna().unique()

    # if graphs are separated by current intervention type
    if args.intervention_type:

        types = df_group["current_intervention_type"].dropna().unique()

        # graph with intervention types as keys and nx graphs as values
        graph_dict = dict.fromkeys(types)

        for itype in types:
            graph_dict[itype] = nx.DiGraph()
            graph_dict[itype].add_nodes_from(guardians)

        for i in tqdm(range(1, df_group.shape[0])):

            guardian_previous = df_group["guardian_id"].values[i - 1]
            guardian_current = df_group["guardian_id"].values[i]

            timestamp_previous = df_group["sent_time"].values[i - 1]
            timestamp_current = df_group["sent_time"].values[i]

            current_type = df_group["current_intervention_type"].values[i]

            if guardian_previous != guardian_current:
                if pd.Timedelta(timestamp_current - timestamp_previous).days == 0:
                    if graph_dict[current_type].has_edge(guardian_previous, guardian_current):
                        graph_dict[current_type][guardian_previous][guardian_current]["weight"] += 1
                    else:
                        graph_dict[current_type].add_edge(guardian_previous, guardian_current, weight=1)

        # construct figure variables
        n_subplots = len(types)

        if n_subplots == 1:
            n_cols = 1
        elif n_subplots == 2:
            n_cols = 2
        else:
            n_cols = 3

        n_rows = n_subplots // n_cols

        if n_subplots % n_cols != 0:
            n_rows += 1

        fig = plt.figure(1)
        fig, ax = plt.subplots(n_rows, n_cols, num=1)
        fig.set_figheight(18 * n_rows)
        fig.set_figwidth(18 * n_cols)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

        plt.box(False)

        for idx, itype in enumerate(types):

            G = graph_dict[itype].copy()

            widths = nx.get_edge_attributes(G, "weight")

            if args.normalized_weights:
                if len(widths) > 0:
                    max_weight = max(widths.values())
                    normalizing_constant = 10 / max_weight
                    widths = {k: v * normalizing_constant for k, v in widths.items()}
            else:
                # threshold to size of width for plots
                widths = {k: min(v, 15) for k, v in widths.items()}

            nodelist = G.nodes()

            ax = fig.add_subplot(n_rows, n_cols, idx + 1)

            pos = nx.shell_layout(G)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=nodelist,
                node_size=600,
                node_color="black",
                alpha=0.7,
                ax=ax,
            )
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=widths.keys(),
                width=list(widths.values()),
                edge_color="lightblue",
                alpha=0.6,
                ax=ax,
            )
            nx.draw_networkx_labels(
                G,
                pos=pos,
                labels=dict(zip(nodelist, nodelist)),
                font_color="white",
                font_size=2,
                ax=ax,
            )
            # ax.box(False)
            ax.axis("off")

            plt.title(
                f"Group {int(gid)} - Guardian sequential interactions graph\nunder {itype} intervention",
                fontsize=30,
            )

        print(f"group_{gid}-normalized_{args.normalized_weights}-guardian_sequential_interactions_graph_by_type.png")
        plt.savefig(
            OUTPUT_PATH
            / f"group_{gid}-normalized_{args.normalized_weights}-guardian_sequential_interactions_graph_by_type.png"
        )
        plt.close()

    # if there is no distinction between interventions
    else:
        G = nx.DiGraph()
        G.add_nodes_from(guardians)

        for i in tqdm(range(1, df_group.shape[0])):

            guardian_previous = df_group["guardian_id"].values[i - 1]
            guardian_current = df_group["guardian_id"].values[i]

            timestamp_previous = df_group["sent_time"].values[i - 1]
            timestamp_current = df_group["sent_time"].values[i]

            if guardian_previous != guardian_current:
                if pd.Timedelta(timestamp_current - timestamp_previous).days == 0:
                    if G.has_edge(guardian_previous, guardian_current):
                        G[guardian_previous][guardian_current]["weight"] += 1
                    else:
                        G.add_edge(guardian_previous, guardian_current, weight=1)

        widths = nx.get_edge_attributes(G, "weight")
        nodelist = G.nodes()

        plt.figure(figsize=(20, 15))

        pos = nx.shell_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=2000, node_color="black", alpha=0.7)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=widths.keys(),
            width=list(widths.values()),
            edge_color="lightblue",
            alpha=0.6,
        )
        nx.draw_networkx_labels(
            G,
            pos=pos,
            labels=dict(zip(nodelist, nodelist)),
            font_color="white",
            font_size=10,
        )
        plt.box(False)

        plt.title(f"Group {int(gid)} - Guardian sequential interactions graph")

        plt.savefig(OUTPUT_PATH / f"group_{gid}-guardian_sequential_interactions_graph.png")
