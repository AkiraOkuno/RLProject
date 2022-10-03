import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pathlib

DATA_PATH = pathlib.Path("data/processed")

# import data
df_guardian = pd.read_csv(DATA_PATH / "df_guardian.csv")
df_moderator = pd.read_csv(DATA_PATH / "df_moderator.csv")
df_interventions = pd.read_csv(DATA_PATH / "df_interventions.csv")
df = pd.read_csv(DATA_PATH / "df_merge.csv")

