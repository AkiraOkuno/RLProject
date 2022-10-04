import argparse
import pathlib
import random
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from tqdm import tqdm


parser = argparse.ArgumentParser()

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

args = parser.parse_args()

DATA_PATH = pathlib.Path("data/processed")

df = pd.read_csv(DATA_PATH / "df_merge.csv")

df["sent_time"] = pd.to_datetime(df["sent_time"])
df["guardian_id"] = df["guardian_id"].astype("Int64")

df["action"] = None
df.loc[~np.isnan(df["interventions_id"].values), "action"] = "intervention"
df.loc[~np.isnan(df["guardian_id"].values), "action"] = "guardian"
df.loc[~np.isnan(df["moderator_id"].values), "action"] = "moderator"

df_guardian = df[df["action"] == "guardian"]

group_ids = df_guardian["groups_id"].dropna().unique()
selected_group_ids = random.sample(list(group_ids), args.random_groups)

for gid in selected_group_ids:

    df_group = df_guardian[df_guardian["groups_id"] == gid]
    guardians = df_group["guardian_id"].dropna().unique()

    G = nx.DiGraph()
    G.add_nodes_from(guardians)

    edgelist = []

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
                # edgelist.append((guardian_previous,guardian_current))

    widths = nx.get_edge_attributes(G, "weight")
    nodelist = G.nodes()

    plt.figure(figsize=(20, 15))

    pos = nx.shell_layout(G)
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist, node_size=2000, node_color="black", alpha=0.7
    )
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
    plt.savefig(f"outputs/plots/group_{gid}-guardian_sequential_interactions_graph.png")
