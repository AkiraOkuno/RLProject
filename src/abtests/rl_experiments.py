import argparse
import json
import os
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from xgboost import XGBRegressor, XGBClassifier
from econml.dr import LinearDRLearner


sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ab_test_number",
    "-ab",
    default=3,
    choices=[1,2,3],
    type = int,
)

args = parser.parse_args()


RAW_PATH = pathlib.Path("data/raw")
DATABASE_PATH = pathlib.Path("outputs/databases")
DATA_PATH = pathlib.Path("data/processed")
OUTPUT_PATH = pathlib.Path("outputs/models/heterogenous_causal_effects")

ORGS1 = [5,6,7,28,29,36,45,46,62]
ORGS2 = [5,6,7,28,29,36,45,46,62]
ORGS3 = [46]

# ab test data
if args.ab_test_number == 1:

    ab_file = 'Noam_Study_1_May26th_TreatmentTags.csv'
    orgs = [5,6,7,28,29,36,45,46,62]
    # date = 26th May 9-11 AM

elif args.ab_test_number == 2:

    ab_file = 'ABTest_2_NoamSheets for uploading - All Arms.csv'
    orgs = [5,6,7,28,29,36,45,46,62]
    # date = 1st August 4-6 PM

else:
    ab_file = 'Nashik_Study_Dropped_Off_Parents_Consolidated_Treatment_List.csv'
    orgs = [46]
    date = datetime(2022, 3, 17, 10) # 17th March 9-10 AM
    # intervention_column = "sample_tag"
    # control_name = "Control - Sample 4"

ab = pd.read_csv(RAW_PATH / ab_file)
ab_guardians = set(ab["guardian_id"].dropna().unique()) 

# intervention data
#interventions = pd.DataFrame(general_utils.open_json(RAW_PATH / "interventions" / "interventions_29.json"))

df_orgs = pd.DataFrame()

for org in orgs:

    df_org = pd.DataFrame(general_utils.open_json(RAW_PATH / "orgs_data" / f"df_guardian_{org}.json"))
    
    df_orgs = pd.concat([df_orgs,df_org])

# preprocess
df_orgs["sent_time"] = pd.to_datetime(df_orgs["sent_time"])

# filter only guardians in ab_test
df_orgs = df_orgs[df_orgs["guardian_id"].isin(ab_guardians)]

# filter only 2 months after intervention
df_orgs = df_orgs[(df_orgs["sent_time"] >= date-timedelta(days=60)) & (df_orgs["sent_time"] <= date+timedelta(days=60))]

# construct variable that assigns to each user if they received treatments
intervention_dict = dict(zip(ab["guardian_id"],ab["sample_tag"]))
df_orgs["intervention"] = df_orgs["guardian_id"].map(intervention_dict)

interventions = list(set(df_orgs["intervention"].unique()) - set(["Control - Sample 4"]))
dummies =  pd.get_dummies(df_orgs["intervention"]).drop(columns=["Control - Sample 4"])
df_orgs = pd.concat([df_orgs, dummies], axis=1)


# build covariates

# open dates guardians entered group json
df_dates = df_org = pd.DataFrame(general_utils.open_json(RAW_PATH / "orgs_data" / f"guardian_date_left_all_orgs.json"))

df_dates = df_dates[df_dates["guardian_id"].isin(ab_guardians)]
df_dates["date_joined"] = pd.to_datetime(df_dates["date_joined"])

date_joined_dict = dict(zip(df_dates["guardian_id"],df_dates["date_joined"]))
df_orgs["date_joined"] = df_orgs["guardian_id"].map(date_joined_dict)

# drop guardians not present in guardian_date_left_all_orgs.json
df_orgs = df_orgs[~df_orgs["date_joined"].isna()]

covariates_dict = dict.fromkeys(df_orgs["guardian_id"].unique())

group_len_dict = df_orgs.groupby("groups_id")["guardian_id"].apply(len).to_dict()
n_messages_by_guardian_dict_before = df_orgs[df_orgs["sent_time"] < date].groupby("guardian_id")["interactions_id"].apply(len).to_dict()
n_messages_by_guardian_dict_after = df_orgs[df_orgs["sent_time"] >= date].groupby("guardian_id")["interactions_id"].apply(len).to_dict()

for guardian in covariates_dict.keys():

    # COVARIATES 

    date_joined = date_joined_dict[guardian]
    n_days_since_joined = (date - date_joined).days

    try:
        n_messages_before = n_messages_by_guardian_dict_before[guardian]
    except KeyError:
        n_messages_before = 0

    group = df_orgs[df_orgs["guardian_id"]==guardian]["groups_id"].values[0]
    group_len = group_len_dict[group]

    guardian_dict = {}
    guardian_dict["n_days_since_joined"] = n_days_since_joined  
    guardian_dict["group_len"] = group_len
    guardian_dict["n_messages_before_intervention"] = n_messages_before

    # TREATMENT
    guardian_dict["intervention"] = ab[ab["guardian_id"]==guardian]["sample_tag"].values[0]

    # OUTCOME
    try:
        guardian_dict["outcome"] = n_messages_by_guardian_dict_after[guardian]
    except KeyError:
        guardian_dict["outcome"] = 0

    covariates_dict[guardian] = guardian_dict


df = pd.DataFrame(covariates_dict).T

dummies =  pd.get_dummies(df["intervention"]).drop(columns=["Control - Sample 4"])
df = pd.concat([df, dummies], axis=1)
#df = df.drop(columns=["intervention"])

# normalize outcome and n_messages_before
#df["n_messages_before_intervention"] /= 60
#df["outcome"] /= 60


# Train EconML model with generic helper models
model = LinearDRLearner(
    model_regression=XGBRegressor(learning_rate=0.1, max_depth=3),
    model_propensity=XGBClassifier(learning_rate=0.1, max_depth=3),
    random_state=1,
)

# Define estimator inputs
T_bin = df[["Treatment - Sample 1",  "Treatment - Sample 2",  "Treatment - Sample 3"]]  # multiple interventions, or treatments
Y = df["outcome"].astype(int)  # amount of product purchased, or outcome
X = df[["n_days_since_joined", "group_len", "n_messages_before_intervention"]]  # heterogeneity feature

# Transform T to one-dimensional array with consecutive integer encoding
def treat_map(t): return np.dot(t, np.arange(1,t.shape[0]+1))

T = np.apply_along_axis(treat_map, 1, T_bin).astype(int)


breakpoint()
# Specify final stage inference type and fit model
model.fit(Y=Y, T=T, X=X, inference="statsmodels")

est = LinearDRLearner()
est.fit(Y.astype(int), T, X=X)
est.effect(X, T0=0, T1=1)

breakpoint()