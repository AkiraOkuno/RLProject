import argparse
import gc
import json
import os
import pathlib
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.utils import general_utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pickle",
    action="store_true",
    help="Save processed data in pickle format",
)
parser.add_argument(
    "--step_1",
    action="store_true",
    help="Execute step 1 of processing",
)
parser.add_argument(
    "--step_2",
    action="store_true",
    help="Execute step 2 of processing",
)
parser.add_argument(
    "--step_3",
    action="store_true",
    help="Execute step 3 of processing",
)
parser.add_argument(
    "--step_4",
    action="store_true",
    help="Execute step 4 of processing = Merge of step 1,2 and 3",
)
args = parser.parse_args()


RAW_PATH = pathlib.Path("data/raw")
DATA_PATH = pathlib.Path("data/processed")
OUTPUT_PATH = pathlib.Path("outputs/databases")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


if args.step_1:

    gc.collect()

    # STEP 1
    ## Create a df1 with fixed guardian and group fixed features
    ## Unique identifiers are guardian_id + group_id
    ## Features:
    ### "n_kids": number of kids a guardian_id has
    ### "male": if there are males in kids set
    ### "female": if there are females in kids set
    ### "n_males": number of males
    ### "n_females": number of females
    ### "private": if any kid went to private preeschool
    ### classes: C1,C2,C3,P1,P2
    ### "n_group_members": number of unique guardian_ids thar ever interacted in group_id

    df = general_utils.open_pickle(DATA_PATH / "df_merge.pickle")

    with open(RAW_PATH / "additional_tables/kids.json", "r") as f:
        df_kids = pd.DataFrame(json.load(f, strict=False))

    # creat a dict that maps group id to number of unique guardian ids that ever interacted ion the group
    # do this before keeping only the valid guardians below
    group_id_to_n_members_dict = dict(
        df[["groups_id", "guardian_id"]].dropna().drop_duplicates().groupby("groups_id")["guardian_id"].nunique()
    )

    # Filter valid guardian_ids
    # which are the ones that don't repeat the same id across different group_ids
    # maybe change this later if more info is given
    # e.g. guardian id 10334 appears in 51 distinct groups ids
    all_guardian_ids = df[["groups_id", "guardian_id"]].dropna().drop_duplicates()["guardian_id"]
    guardian_id_frequency = all_guardian_ids.value_counts()
    valid_guardian_ids = guardian_id_frequency[guardian_id_frequency == 1].index.tolist()

    del guardian_id_frequency

    # create a df that will store the variables to be outputted
    df_out = pd.DataFrame({"guardian_id": valid_guardian_ids})

    # create a dict that maps guardian ids to group ids to complet df_out
    df_ids = df[df["guardian_id"].isin(valid_guardian_ids)][["guardian_id", "groups_id"]].dropna().drop_duplicates()
    guardian_id_to_group_id_dict = dict(zip(df_ids["guardian_id"], df_ids["groups_id"]))
    df_out["group_id"] = df_out["guardian_id"].map(guardian_id_to_group_id_dict)

    del df_ids, guardian_id_to_group_id_dict

    # filter only valid ids
    df = df[df["guardian_id"].isin(valid_guardian_ids)]
    df_kids = df_kids[df_kids["guardian_id"].isin(valid_guardian_ids)]

    # process gender column
    gender_map = {"Female": "F", "Male": "M", "F": "F", "M": "M"}
    df_kids["gender"] = df_kids["gender"].map(gender_map)

    # get kids table feat´ures for each guardian_id
    # as the mapping from kids to guardian_id is many to 1 (multiple kids for 1 guardian), we will
    # create a dict that maps guardian_id to a list of kids, where each element in the list is
    # a dictionary of a kid's features

    guardian_id_to_list_of_kids = {}

    for guardian_id in tqdm(df_kids["guardian_id"].dropna().unique()):

        df_gid = df_kids[df_kids["guardian_id"] == guardian_id]

        # list of dictionaries containing the df features
        guardian_list = df_gid[["gender", "preschool_private", "class"]].to_dict("records")

        guardian_id_to_list_of_kids[guardian_id] = guardian_list

    del df_kids, df_gid

    # merge kids features to main df via guardian_id
    df_out["kids_features"] = df_out["guardian_id"].map(guardian_id_to_list_of_kids)

    # create temporary df that excludes rows without kids_data
    df_temp = df_out.dropna(subset=["kids_features"]).drop(columns=["group_id"])

    # process kids features
    df_temp["n_kids"] = df_temp["kids_features"].apply(lambda kids_list: len(kids_list)).values
    df_temp["n_males"] = df_temp["kids_features"].apply(
        lambda kids_list: len([kid["gender"] for kid in kids_list if kid["gender"] == "M"])
    )
    df_temp["n_females"] = df_temp["kids_features"].apply(
        lambda kids_list: len([kid["gender"] for kid in kids_list if kid["gender"] == "F"])
    )
    df_temp["male"] = (df_temp["n_males"] > 0).astype(int)
    df_temp["female"] = (df_temp["n_females"] > 0).astype(int)

    df_temp["private"] = df_temp["kids_features"].apply(
        lambda kids_list: sum([kid["preschool_private"] for kid in kids_list]) > 0
    )
    df_temp["private"] = df_temp["private"].astype(int)

    df_temp["class"] = df_temp["kids_features"].apply(
        lambda kids_list: tuple(set(([kid["class"] for kid in kids_list if kid["class"] is not None])))
    )

    # df that transforms column class with lists e.g. [p1,p2,c1] to dummies p1,p2,c1,...
    df_class = pd.get_dummies(df_temp["class"].apply(pd.Series).stack()).sum(level=0)

    # merge class dummies to df_temp
    df_temp = df_temp.join(df_class)
    del df_class

    df_temp = df_temp.drop(columns=["kids_features", "class"])

    # merge with df_out
    df_out = df_out.merge(df_temp, on="guardian_id", how="left")
    del df_temp

    df_out = df_out.drop(columns=["kids_features"])

    breakpoint()

    # add group fixed information
    df_out["n_group_members"] = df_out["group_id"].map(group_id_to_n_members_dict)
    del group_id_to_n_members_dict, df

    if args.pickle:
        general_utils.save_pickle(df_out, OUTPUT_PATH / "df1_fixed_guardian_group_features.pickle")

    del df_out

########################################################################################################################################

if args.step_2:

    gc.collect()

    # STEP 2
    ## Create a df2 with daily changing features for group
    ## Unique identifiers are group_id + day
    ## Already have "daily_features_database.pickle" done, which calculates group features at the daily level
    ## so, only have to calculate some additional features and process the data format

    df_daily = general_utils.open_pickle(OUTPUT_PATH / "daily_features_database.pickle")

    # dict to rename variables
    rename = {
        "sent_day": "day",
        "n_guardians": "n_interacting_guardians_daily",
        "n_distinct_moderators": "n_distinct_moderators_daily",
    }
    df_daily = df_daily.rename(rename, axis=1)

    drop = ["guardian_ids", "guardian_ids_history"]
    df_daily = df_daily.drop(columns=drop)

    # make boolean cols become int
    for col in [
        "Other",
        "DA",
        "NR",
        "SV",
        "FM",
        "GroupReportCard",
        "CreateVideoCompilation",
        "ManualCertificate",
        "QF",
    ]:
        df_daily[col] = df_daily[col].astype(int)

    # make date to str
    df_daily["day"] = df_daily["day"].astype(str)

    # calculate DA hour dummies and discretize by 3 hour intervals
    df_hour = pd.get_dummies(df_daily["DA_intervention_hours"].apply(pd.Series).stack()).sum(level=0)

    for h in range(24):
        if h not in df_hour.columns:
            df_hour[h] = 0

    for h in range(8):

        col = f"{3*h}-{3*h+2}"
        df_daily[col] = 0

        for i in range(3):

            if df_hour.shape[0] > 0:

                df_daily[col] += df_hour[3 * h + i]

                # fill nas with 0, i.e. all hour values are 0 for days without interventions
                df_daily[col] = df_daily[col].fillna(0)

            else:
                # if there is no intervention in the whole period
                df_daily[col] = 0

        # map non zeroes to 1
        df_daily[col] = df_daily[col].astype(bool).astype(int)

    df_daily = df_daily.drop(columns=["DA_intervention_hours"])

    # for the follwing columns, remove duplicates in lists and make dummy columns
    cols = ["language", "activity_type", "difficulty_level", "response_type", "audience", "learning_domain"]

    # before doing the operations, substitute "other" value of "response type" as there is already an "other" column of intervention type
    # Other -> OtherResponseType
    df_daily["response_type"] = df_daily["response_type"].apply(
        lambda x: ["OtherResponseType" if el == "Other" else el for el in x]
    )

    # substitute "other" value of "learning_domain" as there is already an "other" column of intervention type
    # Other -> OtherLearningDomain
    df_daily["learning_domain"] = df_daily["learning_domain"].apply(
        lambda x: ["OtherLearningDomain" if el == "other" else el for el in x]
    )

    for col in cols:

        # remove dups in lists
        df_daily[col] = df_daily[col].apply(lambda x: list(set(x)))

        # make dummies and join
        df_col = pd.get_dummies(df_daily[col].apply(pd.Series).stack()).sum(level=0)
        df_daily = df_daily.join(df_col)

    # drop original columns
    df_daily = df_daily.drop(columns=cols)

    # add modetator messages length and type of mod message (eg video, audio, etc)
    df = general_utils.open_pickle(DATA_PATH / "df_merge.pickle")

    # message type
    df_message_type = (
        df[df["moderator_id"].notna()].groupby(["groups_id", "sent_date"])["message_type"].unique().reset_index()
    )
    df_message_type = df_message_type.rename({"groups_id": "group_id", "sent_date": "day"}, axis=1)

    df_daily = df_daily.merge(df_message_type, on=["group_id", "day"], how="left")
    del df_message_type

    # substitue none by empty list in message type column
    df_daily["message_type"] = df_daily["message_type"].fillna("").apply(list)

    # add ModMessageType to message type strings, so that the columns dummies are identifiable
    df_daily["message_type"] = df_daily["message_type"].apply(
        lambda x: ["ModMessageType" + el for el in x if el is not None]
    )

    # make dummy
    df_message_type = pd.get_dummies(df_daily["message_type"].apply(pd.Series).stack()).sum(level=0)
    df_daily = df_daily.join(df_message_type)
    del df_message_type

    # length of moderator messages in a day, in a given group
    df_length = (
        df[df["moderator_id"].notna()]
        .dropna(subset=["text"])
        .groupby(["groups_id", "sent_date"])["text"]
        .apply(lambda x: len(" ".join(x)))
        .reset_index()
    )
    df_length = df_length.rename(
        {"groups_id": "group_id", "sent_date": "day", "text": "mod_message_length_daily"}, axis=1
    )
    df_daily = df_daily.merge(df_length, on=["group_id", "day"], how="left")

    del df_length, df

    # TODO: add past group interaction features, e.g. number of messages in last month, active parents last week

    if args.pickle:
        general_utils.save_pickle(df_daily, OUTPUT_PATH / "df2_daily_group_features.pickle")

    del df_daily

########################################################################################################################################

if args.step_3:

    gc.collect()

    # STEP 3
    ## Create a df3 with daily changing features for guardians in a group
    ## Unique identifiers are guardian_id + group_id + day

    df = general_utils.open_pickle(DATA_PATH / "df_merge.pickle")

    # Filter valid guardian_ids
    all_guardian_ids = df[["groups_id", "guardian_id"]].dropna().drop_duplicates()["guardian_id"]
    guardian_id_frequency = all_guardian_ids.value_counts()
    valid_guardian_ids = guardian_id_frequency[guardian_id_frequency == 1].index.tolist()

    df = df[df["guardian_id"].isin(valid_guardian_ids)]

    # dict that maps group id to list of guardian ids
    # group_id_to_list_guardian_ids = dict(df[["groups_id","guardian_id"]].dropna().groupby("groups_id")["guardian_id"].unique())

    # dict that maps guardian id to respective group id
    # df_gg = df[["groups_id","guardian_id"]].dropna()
    # group_id_to_list_guardian_ids = dict(zip(df_gg["guardian_id"],df_gg["groups_id"]))
    # del df_gg

    # relates guardian id and day to number of interactions of guardian at that day
    df_guardian_activity = (
        df.groupby(["guardian_id", "groups_id", "sent_date"])["interactions_id"].count().reset_index()
    )

    df_guardian_activity = df_guardian_activity.sort_values("sent_date")
    df_guardian_activity["sent_date"] = pd.to_datetime(df_guardian_activity["sent_date"])

    df_out = pd.DataFrame()

    for guardian_id in tqdm(valid_guardian_ids):

        dfg_day = df_guardian_activity[df_guardian_activity["guardian_id"] == guardian_id]

        group_id = dfg_day["groups_id"].values[0]

        min_date = dfg_day["sent_date"].values[0]  # guardian first interacted
        max_date = df["sent_date"].values[-1]  # last day of data

        # fills daily missing values with 0
        full_index = pd.date_range(min_date, max_date)

        interaction_series = dfg_day.set_index("sent_date")["interactions_id"]
        interaction_series.index = pd.DatetimeIndex(interaction_series.index)

        interaction_series = interaction_series.reindex(full_index)

        dfg_day = dfg_day.set_index("sent_date")["interactions_id"].reindex(full_index, fill_value=0).reset_index()
        dfg_day = dfg_day.rename(
            {"index": "day", "interactions_id": "n_individual_guardian_interactions_daily"}, axis=1
        )

        dfg_day["guardian_id"] = guardian_id
        dfg_day["group_id"] = group_id

        # calculate rolling variables, e.g. rolling sum of week number of interactions
        dfg_day["week_cumulative_n_guardian_messages"] = (
            dfg_day["n_individual_guardian_interactions_daily"].rolling(8, min_periods=1).sum()
        )

        # month cumulative messages
        dfg_day["month_cumulative_n_guardian_messages"] = (
            dfg_day["n_individual_guardian_interactions_daily"].rolling(31, min_periods=1).sum()
        )

        # create temp column for indicator if guardian interacted in the day
        dfg_day["guardian_interacted"] = (dfg_day["n_individual_guardian_interactions_daily"] > 0).astype(int)

        # weekly cumulative days of interactions
        dfg_day["weekly_cumulative_guardian_n_days_interacted"] = (
            dfg_day["guardian_interacted"].rolling(7, min_periods=1).sum().fillna(0)
        )

        # monthly cumulative days of interactions
        dfg_day["monthly_cumulative_guardian_n_days_interacted"] = (
            dfg_day["guardian_interacted"].rolling(30, min_periods=1).sum().fillna(0)
        )

        # exclude current day form cumulative counts
        dfg_day["weekly_cumulative_guardian_n_days_interacted"] -= dfg_day["guardian_interacted"]
        dfg_day["monthly_cumulative_guardian_n_days_interacted"] -= dfg_day["guardian_interacted"]

        # indicator: guardian has interacted in the last week
        dfg_day["weekly_guardian_interaction_indicator"] = (
            dfg_day["weekly_cumulative_guardian_n_days_interacted"] > 0
        ).astype(int)

        # indicator: guardian has interacted in the last month
        dfg_day["monthly_guardian_interaction_indicator"] = (
            dfg_day["monthly_cumulative_guardian_n_days_interacted"] > 0
        ).astype(int)

        df_out = pd.concat([df_out, dfg_day])

    df_out = df_out.reset_index(drop=True)

    df_out["day"] = df_out["day"].astype(str)

    # TODO: remove dates/rows for guardians that are more than 4 months without interacting and dont ever interact again in the data = disengaged
    # maybe consider max_date = dfg_day["sent_date"].values[-1]
    # maybe create a 4 months indicator variable to filter these disengaged users

    if args.pickle:
        general_utils.save_pickle(df_out, OUTPUT_PATH / "df3_daily_guardian_features.pickle")

    del df, df_out, dfg_day, full_index, interaction_series


########################################################################################################################################

if args.step_4:

    gc.collect()

    # unique identifiers: guardian_id
    # fixed guardian/group features
    df1 = general_utils.open_pickle(OUTPUT_PATH / "df1_fixed_guardian_group_features.pickle")

    # unique identifiers: day + group_id
    # daily group features + intervention at the group features
    df2 = general_utils.open_pickle(OUTPUT_PATH / "df2_daily_group_features.pickle")

    df = df1.merge(df2, on="group_id", how="left")
    del df1, df2

    # remove rows with na's
    # MAYBE change this later! but still a lot of data...
    df = df.dropna()

    # unique identifiers: day + guardian_id
    df3 = general_utils.open_pickle(OUTPUT_PATH / "df3_daily_guardian_features.pickle")
    gc.collect()

    df = df.merge(df3, on=["day", "guardian_id", "group_id"], how="left")
    del df3

    # fine tuning of preprocessing
    df["Household Objects"] += df["Household objects"]
    df = df.drop(columns=["Household objects", "message_type"])

    df_weekday = pd.get_dummies(df["weekday"], drop_first=True)
    df_weekday.columns = [f"weekday-{col}" for col in df_weekday.columns]
    df = df.drop(columns=["weekday"])
    df = df.join(df_weekday)

    df.to_csv(OUTPUT_PATH / "df4_features_merge.csv")

    if args.pickle:
        general_utils.save_pickle(df, OUTPUT_PATH / "df4_features_merge.pickle")

    # MAYBE change this later! but still a lot of data...
    df = df.dropna()

    if args.pickle:
        general_utils.save_pickle(df, OUTPUT_PATH / "df4_features_merge_without_nans.pickle")
