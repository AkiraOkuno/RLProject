import argparse
import os
import pathlib
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--group_fixed_effects",
    "-fe",
    action="store_true",
    help="Add group_fixed effects to covariate table",
)
args = parser.parse_args()


DATA_PATH = pathlib.Path("data/processed")
DATABASES_PATH = pathlib.Path("outputs/databases")

OUTPUT_PATH = pathlib.Path("outputs/plots/featurized_simulation")
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

breakpoint()
df = df.drop(columns=always_drop)

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
    "week_cumulative_n_guardian_messages",
    "month_cumulative_n_guardian_messages",
    "Easy",
    "ModMessageTypeChat",
    "weekly_guardian_interaction_indicator",
    "monthly_guardian_interaction_indicator",
    "n_individual_guardian_interactions_daily",
    "monthly_cumulative_guardian_n_days_interacted",
    # "n_group_members",
    "n_group_members_sq",
]

X = df.drop(columns=drop)
X = X.drop(columns=drop2)

model = LogisticRegression(solver="liblinear", random_state=0)
model.fit(X, y)

log_reg = sm.Logit(y, X).fit_regularized(method="l1")
print(log_reg.summary())
breakpoint()

import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)
predicted_y = model.predict(X_test)
print(metrics.classification_report(y_test, predicted_y))
# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)


def feat_imp(df, model):

    d = dict(zip(df.columns, model.feature_importances_))
    ss = sorted(d, key=d.get, reverse=True)
    n_features = len(ss)
    top_names = ss[0:]

    plt.figure(figsize=(25, 15))
    plt.title("Feature importances")
    plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
    # plt.xlim(-1, n_features)
    plt.xticks(range(n_features), top_names, rotation="vertical")
    plt.savefig("xgb_feature_importance.png")
    plt.tight_layout()

    plt.close()


feat_imp(X_train, model)
breakpoint()

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
x = X.iloc[:, :]
vif_data["feature"] = x.columns
vif_data["VIF"] = [variance_inflation_factor(x.values, i) for i in range(len(x.columns))]

breakpoint()
