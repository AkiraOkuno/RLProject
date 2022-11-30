import argparse
import itertools
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.manifold import TSNE
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--calculate_matrix",
    "-c",
    help="whether to calculate the response matrix",
    action="store_true",
)

args = parser.parse_args()

DATA_PATH = pathlib.Path("outputs/databases")

OUTPUT_PATH = pathlib.Path("outputs/models/svd_results")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

if args.calculate_matrix:

    df = general_utils.open_pickle(DATA_PATH / "daily_features_database.pickle")

    # ONLY select groups with more than 180 days of observations
    valid_groups = df["group_id"].dropna().value_counts()
    valid_groups = valid_groups[valid_groups > 180].index.values

    intervention_types = [
        "Other",
        "DA",
        "NR",
        "SV",
        "FM",
        "GroupReportCard",
        "CreateVideoCompilation",
        "ManualCertificate",
        "QF",
    ]

    response_matrix = []

    for group in tqdm(valid_groups):

        dfg = df[df["group_id"] == group]

        guardian_ids = sorted(set(itertools.chain.from_iterable(dfg["guardian_ids"].dropna())))

        group_matrix = np.zeros([len(intervention_types), len(guardian_ids)])

        for i, itype in enumerate(intervention_types):

            df_active_intervention = dfg[dfg[itype]]
            n_days_under_intervention = df_active_intervention.shape[0]

            for j, gid in enumerate(guardian_ids):

                filter = df_active_intervention["guardian_ids"].explode() == gid
                df_guardian = df_active_intervention.loc[filter[filter].index]

                n_days_active_guardian_under_intervention = df_guardian.shape[0]

                # prob_itype_gid = n_days_active_guardian_under_intervention / n_days_under_intervention
                response_matrix.append(
                    [gid, group, itype, n_days_active_guardian_under_intervention, n_days_under_intervention]
                )

    response_matrix = pd.DataFrame(response_matrix)
    response_matrix.columns = ["guardian_id", "group_id", "intervention_type", "active_days", "total_days"]

    # unique id = guardian_id+group_id == ggid
    response_matrix["ggid"] = response_matrix["guardian_id"].astype(str) + "-" + response_matrix["group_id"].astype(str)

    response_matrix["response_rate"] = response_matrix["active_days"] / response_matrix["total_days"]

    # drop nas = division by 0
    response_matrix = response_matrix.dropna()

    # save matrix
    general_utils.save_pickle(response_matrix, DATA_PATH / "response_matrix.pickle")

else:

    response_matrix = general_utils.open_pickle(DATA_PATH / "response_matrix.pickle")

# filter non active users in response matrix
valid_users = response_matrix.groupby(["guardian_id", "group_id"])["response_rate"].sum().reset_index()
threshold = valid_users["response_rate"].quantile(0.7)
valid_users = valid_users[valid_users["response_rate"] > threshold]
valid_users["ggid"] = valid_users["guardian_id"].astype(str) + "-" + valid_users["group_id"].astype(str)
valid_users = valid_users["ggid"].dropna().unique()

breakpoint()

# filter response matrix
response_matrix = response_matrix[response_matrix["ggid"].isin(valid_users)]

# create surprise dataset
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(
    response_matrix[response_matrix["intervention_type"] != "DA"][["ggid", "intervention_type", "response_rate"]],
    reader,
)
svd = SVD(verbose=True, n_epochs=100, n_factors=2, biased=False)
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

print(max([svd.predict(x[0], x[1]).est for x in svd.trainset.build_testset()]))
print("----------")

results = pd.DataFrame(svd.pu)
results.columns = ["d1", "d2"]
ids = [svd.trainset.to_raw_uid(x) for x in range(svd.pu.shape[0])]
results["id"] = ids
results = results.set_index("id")

for iid in svd.trainset.all_items():
    results[svd.trainset.to_raw_iid(iid)] = 0

for uid in svd.trainset.all_users():

    raw_uid = svd.trainset.to_raw_uid(uid)

    user_dict = svd.trainset.ur[uid]

    for iid, response in user_dict:
        raw_iid = svd.trainset.to_raw_iid(iid)
        results.loc[raw_uid, raw_iid] = response

# do not include DA
for intervention in results.columns[2:]:

    # plot factors
    plt.figure(figsize=(20, 20))
    fig = px.scatter(results, x="d1", y="d2", color=intervention, width=1000, size=intervention, height=1000)
    # fig.update_traces(marker_size=2.5)
    fig.write_image(OUTPUT_PATH / f"user_vectors_{intervention}.png")
    plt.close()

# sum over all intervention types
results["sum"] = results.iloc[:, 2:].sum(axis=1)
plt.figure(figsize=(20, 20))
fig = px.scatter(results, x="d1", y="d2", color="sum", width=1000, size="sum", height=1000)
fig.write_image(OUTPUT_PATH / "user_vectors_sum.png")
plt.close()

# plot product features
product_df = pd.DataFrame(svd.qi)
product_df.columns = ["d1", "d2"]
plt.figure(figsize=(20, 20))
fig = px.scatter(product_df, x="d1", y="d2", width=500, height=500)
fig.write_image(OUTPUT_PATH / "product_vectors.png")
plt.close()
breakpoint()

# sim = svd.compute_similarities()
# ids = [svd.trainset.to_raw_uid(x) for x in range(svd.pu.shape[0])]
# sns.heatmap(sim, center=0, cmap="vlag", linewidths=0.75, square=True, vmin=0, vmax=1)

# histogram of user fixed coefficients
plt.hist(svd.bu, bins=100)
plt.savefig(OUTPUT_PATH / "user_coef_hist_DA.png")
plt.close()

# reestimate SVD without DA
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(
    response_matrix[response_matrix["intervention_type"] != "DA"][["ggid", "intervention_type", "response_rate"]],
    reader,
)
svd = SVD(verbose=True, n_epochs=50, n_factors=2)
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# plot factors
fig = px.scatter(x=svd.pu[:, 0], y=svd.pu[:, 1])
fig.write_image(OUTPUT_PATH / "user_vectors_no_DA.png")

# histogram of user fixed coefficients
plt.hist(svd.bu, bins=100)
plt.savefig(OUTPUT_PATH / "user_coef_hist_no_DA.png")
plt.close()

breakpoint()

tsne = TSNE(n_components=2, n_iter=250, verbose=3, random_state=1)
guardian_embedding = tsne.fit_transform(svd.pu)
projection = pd.DataFrame(columns=["x", "y"], data=guardian_embedding)
projection["id"] = response_matrix["ggid"]

fig = px.scatter(projection, x="x", y="y")
fig.show()
