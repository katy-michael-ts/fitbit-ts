# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Fitbit Time Series Project
# By Kathryn Salts & Michael Moran

# ## TODO

# ## Table of contents
# 1. [Project Planning](#project-planning)
# 1. [Acquisition](#acquisition)
# 1. [Preparation](#preparation)
# 1. [Exploration](#exploration)
# 1. [Modeling](#modeling)

# ## Project Planning <a name="project-planning"></a>

# ### Goals

# ### Deliverables

# ### Data Dictionary & Domain Knowledge

# ### Hypotheses

# ### Thoughts & Questions

# ### Prepare the Environment

# +
import os
from pprint import pprint
from enum import Enum, auto
import io
from datetime import timedelta

import pandas as pd


# -

# **Reload modules to capture changes**

# ## Acquisition <a name="acquisition"></a>

class CSVType(Enum):
    CALS_IN = auto()
    ACTIVITIES = auto()
    FOOD_LOG = auto()


# +
DATADIR = "fitbit"

CALS_IN_HEADER = "Date,Calories In"
ACTIVITIES_HEADER = "Date,Calories Burned,Steps,Distance,Floors,Minutes Sedentary,Minutes Lightly Active,Minutes Fairly Active,Minutes Very Active,Activity Calories"
FOOD_LOG_HEADER = "Daily Totals"

cals_in = []
activities = []
food_log = []
for file in sorted(os.listdir(DATADIR)):
    with open(DATADIR + '/' + file) as fp:
        # print(f"Reading {file}")
        lines = fp.readlines()
        csvtype = None
        date = ""
        for line in lines:
            # print("Line:", lines)
            
            # START OF NEW CSV
            if line.startswith("Foods"):
                csvtype = CSVType.CALS_IN 
            elif line.startswith("Activities"):
                csvtype = CSVType.ACTIVITIES 
            elif line.startswith("Food Log"):
                toks = line.split()
                date = toks[2]
                csvtype = CSVType.FOOD_LOG
                
            # START OF HEADER
            elif line.startswith(CALS_IN_HEADER) or line.startswith(ACTIVITIES_HEADER) or\
            line.startswith(FOOD_LOG_HEADER):
                continue
            
            # IT'S DATA!
            elif csvtype == CSVType.CALS_IN:
                cals_in.append(line)
            elif csvtype == CSVType.ACTIVITIES:
                activities.append(line)
            elif csvtype == CSVType.FOOD_LOG:
                if line.startswith('""'):
                    line = line.replace('""', f'"{date[:4]}-{date[4:6]}-{date[6:]}"')
                    print(repr(line))
                food_log.append(line)

# -

pprint(cals_in)

pprint(activities)

pprint(food_log)

cals_in_cols = ["date", "calories_in"]
df_cals_in = pd.read_csv(io.StringIO("".join(cals_in)), header=None, names=cals_in_cols)

df_cals_in.info()

df_cals_in.sample(n=10)

df_cals_in.T


# +
def df_col_to_datetime(df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
    datetimed = pd.to_datetime(df[col], **kwargs)
    out = df.copy()
    out[col] = datetimed
    return out

def series_remove_commas(series: pd.Series) -> pd.Series:
    return series.str.replace(",", "")

def df_cals_in_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df["calories_in"] = series_remove_commas(df.calories_in)
    df = df_col_to_datetime(df_cals_in, "date", format="%Y-%m-%d")
    return df.astype({"calories_in": int})


# -

df_cals_in = df_cals_in_prepare(df_cals_in)

df_cals_in.info()

df_cals_in.sample(n=10)


# +
def has_every_day(date_series: pd.Series) -> bool:
    return timedelta(date_series.nunique()) == (date_series.max() - date_series.min()) + timedelta(days=1)

has_every_day(df_cals_in.date)
# -

activities_cols = ["date", "calories_burned", "steps", "distance", 
              "floors", "minutes_sedentary", "minutes_lightly_active", 
              "minutes_fairly_active", "minutes_very_active", "activity_calories"]
df_activities = pd.read_csv(io.StringIO("".join(activities)), header=None, names=activities_cols)

df_activities.info()

df_activities.sample(n=10)


def df_activities_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df_col_to_datetime(df, "date", format="%Y-%m-%d")
    
    df["calories_burned"] = series_remove_commas(df.calories_burned)
    df["steps"] = series_remove_commas(df.steps)
    df["minutes_sedentary"] = series_remove_commas(df.minutes_sedentary)
    df["activity_calories"] = series_remove_commas(df.activity_calories)
    
    return df.astype({"calories_burned": int, "steps": int, "minutes_sedentary": int, "activity_calories": int})


df_activities_prepare(df_activities).info()

df_activities_prepare(df_activities).sample(n=10)

# food_log_cols = ["date", "calories", "fat", "fiber", "carbs", "sodium",
#                  "protein", "water"]
food_log_cols = ["date", "column", "value"]
df_food_log = pd.read_csv(io.StringIO("".join(food_log)), header=None, names=food_log_cols)

df_food_log.info()

df_food_log.head(10)

df_food_log.groupby("date")["column", "value"]

df_food_log.set_index(["date", "column"]).unstack(level=-1)

df_food_log.column.value_counts(dropna=False)

# **I need to iterate over the DataFrame, use the date as the index, make columns for calories, fat, etc and put the value there**

df = pd.DataFrame({"col1": [1,1,1,1], "col2": [10,11,12,13], "col3": [99, 98, 97, 96]})

df.head()

df.set_index(["col1", "col2"]).unstack(level=-1)


# +
def df_food_log_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df_col_to_datetime(df, "date", format="%Y-%m-%d")
    
#     df["calories_burned"] = series_remove_commas(df.calories_burned)
#     df["steps"] = series_remove_commas(df.steps)
#     df["minutes_sedentary"] = series_remove_commas(df.minutes_sedentary)
#     df["activity_calories"] = series_remove_commas(df.activity_calories)
    
#     return df.astype({"calories_burned": int, "steps": int, "minutes_sedentary": int, "activity_calories": int})
# -

# ## Preparation <a name="preparation"></a>

# ### Summarize Data

# ### Handle Missing Values

# ### Handle Duplicates

# ### Fix Data Types

# ### Handle Outliers

# ### Check Missing Values

# ## Exploration  <a name="exploration"></a>

# ### Train-Test Split

# ### Visualizations

# ### Statistical Tests

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

# ### Feature Engineering & Selection

# ### Train & Test Models

# ### Summarize Conclusions
