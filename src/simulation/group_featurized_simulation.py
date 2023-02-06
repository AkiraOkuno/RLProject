import argparse
import os
import pathlib
import random
import sys
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from xgboost import plot_importance

sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--group_fixed_effects",
    "-fe",
    action="store_true",
    help="Add group_fixed effects to covariate table",
)
parser.add_argument(
    "--n_test_splits",
    "-nt",
    type=int,
    help="Number of temporal splits for XGB cross validation",
)
parser.add_argument(
    "--pickle",
    action="store_true",
    help="Whether to save final processed database pickle",
)
args = parser.parse_args()


DATA_PATH = pathlib.Path("data/processed")
DATABASES_PATH = pathlib.Path("outputs/databases")

OUTPUT_PATH = pathlib.Path("outputs/models/featurized_model")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

df = general_utils.open_pickle(DATABASES_PATH / "df4_features_merge_without_nans.pickle")

# GENERAL FILTERS
df = df[df["n_group_members"] > 1]

# remove private preschool kids
df = df[df["private"] == 0]
df = df.drop(columns=["private"], axis=1)

# only parents with up to 5 kids, 108.9k -> 102.7k obs
df = df[df["n_kids"] <= 5]

if args.group_fixed_effects:

    # remove groups with very low value counts (below 100), 102.7k -> 95.7k
    # mean group count after removal is 490 and there are 195 groups
    # will be used only if create group FE
    group_count = df["group_id"].value_counts()
    valid_group_ids = set(group_count[group_count >= 400].index.values)

    df = df[df["group_id"].isin(valid_group_ids)]
    del group_count, valid_group_ids

# CREATE NEW FEATURES

if args.group_fixed_effects:

    # group FE
    df_fe = pd.get_dummies(df["group_id"], drop_first=True)
    df_fe.columns = [f"Group-fe-{group_id}" for group_id in df_fe.columns]
    df = df.join(df_fe)
    del df_fe

# number of static members squared
df["n_group_members_sq"] = df["n_group_members"] ** 2

# proportion of guardians in the group that interacted in a day
df["proportion_guardians_interacted_daily"] = df["n_interacting_guardians_daily"] / df["n_group_members"]

# whether there are kids in preprimary
df["Preprimary_class"] = (df["P1"] + df["P2"] > 0).astype(int)
df["Primary_class"] = (df["C1"] + df["C2"] + df["C3"] > 0).astype(int)

# At least 1 kid in preprimary and no other kid in primary, and vice-versa: both sum up to 95% of observations
df["Only_preprimary"] = ((df["Preprimary_class"] == 1) & (df["Primary_class"] == 0)).astype(int)
df["Only_primary"] = ((df["Preprimary_class"] == 0) & (df["Primary_class"] == 1)).astype(int)

# Activity is focused for any primary class / preprimary class
df["Primary_Activity"] = (
    df["Primary I (6-7 years)"] + df["Both(Primary I (6-7 years) and Primary II (7-8 years))"] > 0
).astype(int)
df["Preprimary_Activity"] = (
    df["Both(Pre-Primary I (3-4 years) and Pre-Primary II (5-6 years))"]
    + df["Pre-Primary I (3-4 years)"]
    + df["Pre-Primary II (5-6 years)"]
    > 0
).astype(int)

# activity class == kid's class (kids class: test both "only_" and "_class")
df["Activity_matching_class"] = (
    ((df["Primary_Activity"] == 1) & (df["Only_primary"] == 1))
    | ((df["Preprimary_Activity"] == 1) & (df["Only_preprimary"] == 1))
).astype(int)

# guardian has at least one male and no female, and vice versa: sum up to 95%
df["Only_male"] = ((df["male"] == 1) & (df["female"] == 0)).astype(int)  # 45%
df["Only_female"] = ((df["male"] == 0) & (df["female"] == 1)).astype(int)  # 49%

# total number of nudges
nudge_cols = ["Other", "NR", "SV", "FM", "GroupReportCard", "CreateVideoCompilation", "ManualCertificate", "QF"]
df["n_nudges"] = 0

for col in nudge_cols:
    df["n_nudges"] += df[col]

# number of nudges relative to group size
df["nudges_per_group_size"] = df["n_nudges"] / df["n_group_members"]

# sort by dates
df = df.sort_values("day").reset_index(drop=True)

# save indices for temporal test split
min_date = datetime.strptime(df["day"].min(), "%Y-%m-%d")
max_date = datetime.strptime(df["day"].max(), "%Y-%m-%d")
date_range_days = (max_date - min_date).days

# create output variable: daily binary response
y = df["guardian_interacted"]

# list of columns that have to be always dropped
always_drop = [
    "guardian_interacted",
    "guardian_id",
    "group_id",
    "day",
    "DA",
    "hi",
    "Activity",
    "OtherResponseType",
    "text",
    "ModMessageTypeSticker",
    "0-2",
    "n_nudges",
]

# save day column values
days_series = df["day"]

X = df.drop(columns=always_drop)

# list of features to be dropped, not to be fixed, used for test
# drop: most likely to be officially dropped
# drop2: more prone to testing

drop = [
    "P1",
    "P2",
    "C1",
    "C2",
    "C3",
    "Primary I (6-7 years)",
    "Both(Primary I (6-7 years) and Primary II (7-8 years))",
    "Both(Pre-Primary I (3-4 years) and Pre-Primary II (5-6 years))",
    "Pre-Primary I (3-4 years)",
    "Pre-Primary II (5-6 years)",
]

drop2 = [
    "male",
    "female",
    "n_males",
    "n_females",
    "Preprimary_class",
    "Primary_class",
    "mr",
    "n_distinct_moderators_daily",
    "Easy",
    "ModMessageTypeChat",
    "n_individual_guardian_interactions_daily",
    # "n_group_members",
    "n_group_members_sq",
    "week_cumulative_n_guardian_messages",
    "month_cumulative_n_guardian_messages",
    "weekly_guardian_interaction_indicator",
    "monthly_guardian_interaction_indicator",
    "weekly_cumulative_guardian_n_days_interacted",
    "monthly_cumulative_guardian_n_days_interacted",
    # "weekly_cumulative_guardian_n_days_interacted_lag_1d",
    "monthly_cumulative_guardian_n_days_interacted_lag_1d",
    # "weekly_cumulative_guardian_n_days_interacted_lag_1w",
    "monthly_cumulative_guardian_n_days_interacted_lag_1w",
    # "weekly_guardian_interaction_indicator_lag_1d",
    "monthly_guardian_interaction_indicator_lag_1d",
    # "weekly_guardian_interaction_indicator_lag_1w",
    "monthly_guardian_interaction_indicator_lag_1w",
    "guardian_interaction_next_60_days",
    "guardian_interaction_next_120_days",
]


X = X.drop(columns=drop)
X = X.drop(columns=drop2)

if args.pickle:

    data = (X, y)
    general_utils.save_pickle(data, path=DATABASES_PATH / "featurized_training_data.pickle")

model = LogisticRegression(solver="liblinear", random_state=0)
model.fit(X, y)

log_reg = sm.Logit(y, X).fit_regularized(method="l1")
summary = log_reg.summary()
print(summary)

# save logreg summary table
general_utils.save_pickle(summary, path=OUTPUT_PATH / "logreg_summary_table.pickle")

#########################################

# run xgboost classification with temporal validation

# re-add day column
X["day"] = days_series
X["day"] = pd.to_datetime(X["day"])

# number of temporal splits
n_splits = args.n_test_splits

# if there are 600 days and 10 splits, we want days_split = [0,60,120,180,...,600]
days_split = list(range(0, date_range_days, int(date_range_days / (n_splits + 1))))
days_split = days_split[:-1]
days_split.append(date_range_days + 1)

# convert to actual datetime
days_split = [min_date + timedelta(days=x) for x in days_split]

results = []

# run each epoch with cumulative training sets
for epoch in tqdm(range(1, n_splits + 1)):

    # date of last obs. in current epoch training data
    upper_date = days_split[epoch]

    # date of last obd. in current epoch test data
    test_date = days_split[epoch + 1]

    # create train and test sets
    X_train = X[X["day"] <= upper_date].drop(columns=["day"])
    X_test = X[(X["day"] > upper_date) & (X["day"] <= test_date)].drop(columns=["day"])

    train_upper_index = X_train.index.values[-1]
    test_upper_index = X_test.index.values[-1]

    y_train = y[y.index <= train_upper_index]
    y_test = y[(y.index > train_upper_index) & (y.index <= test_upper_index)]

    # run model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    predicted_y = model.predict(X_test)

    # report metrics
    report = metrics.classification_report(y_test, predicted_y)

    precision = metrics.precision_score(y_test, predicted_y)
    recall = metrics.recall_score(y_test, predicted_y)
    accuracy = metrics.accuracy_score(y_test, predicted_y)

    results.append((precision, recall, accuracy))

# create df for performance time series
df_results = pd.DataFrame(results)
df_results.columns = ["precision", "recall", "accuracy"]

print(df_results)


# def feat_imp(df, model):

#     d = dict(zip(df.columns, model.feature_importances_))
#     ss = sorted(d, key=d.get, reverse=True)
#     n_features = len(ss)
#     top_names = ss[0:]

#     plt.figure(figsize=(25, 15))
#     plt.title("Feature importances")
#     plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
#     # plt.xlim(-1, n_features)
#     plt.xticks(range(n_features), top_names, rotation="vertical")
#     plt.savefig("xgb_feature_importance.png")
#     plt.tight_layout()

#     plt.close()


# feat_imp(X_train, model)
# breakpoint()

# from statsmodels.stats.outliers_influence import variance_inflation_factor

# vif_data = pd.DataFrame()
# x = X.iloc[:, :]
# vif_data["feature"] = x.columns
# vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

# breakpoint()
