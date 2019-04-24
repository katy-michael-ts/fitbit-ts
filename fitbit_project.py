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
#
# - [ ] [What to do with extra data?](#todo-extra-data)

# ## Table of contents
# 1. [Project Planning](#project-planning)
# 1. [Acquisition](#acquisition)
# 1. [Preparation](#preparation)
# 1. [Exploration](#exploration)
# 1. [Modeling](#modeling)

# ## Project Planning <a name="project-planning"></a>

# ### Goals

# 1. Identify who was wearing the Fitbit
# 1. Predict what next two weeks of data will be

# ### Deliverables

# 1. CSV with predictions of next two weeks of data
# 1. Two slides describing the Fitbit data and our predictions. Must contain at least one visualization
# 1. Jupyter notebook showing work
# 1. Tidy dataset

# ### Data Dictionary & Domain Knowledge
#
# - Caloric burn per day
#     - avg man: 2,200 - 3,000
#     - avg woman: 1,800 - 2,2000
#     - [Source](https://www.livestrong.com/article/278257-how-many-calories-does-the-body-naturally-burn-per-day/)

# ### Hypotheses
#
# 1. The wearer may be testing fitness trackers.
# 1. The wearer is fairly active, but not mostly mobile activity, possibly stationary like lifting weights.
# 1. Wearer is not someone in a drug trial because there are not many entries in the food log. We would expect them to be logging food to see if there are any interactions.
# 1. The wearer is likely not wearing the tracker while sleeping. Average inactivity/activity minutes is 16-17 hours per day.
#     - Wouldn't expect a person testing fitness equipment to wear it that long
#     - Would make sense for an employee or drug trial participant (but we would expect the drug trial participant to wear it while sleeping)
# 1. Likely not a person testing fitness equipment because there are food log entries for one week; and also likely not a drug trial participant for the same reason
#     - Makes it more likely to be an employee who lost the motivation to log food
# 1. Looks like person stopped wearing tracker on 12/7/18 because there are food log entries and caloric intake entries for dates after the 7th, but the activity log stops on the 6th.

# ### Thoughts & Questions
#
# 1. What does the weekend data look like? This may tell us whether they work there or are in a drug trial  (likely to wear the fitbit on the weekend) or are testing fitness equipment (not likely to wear on weekend)
# 1. Why does the activity tracking data end on 12/7 but the food log and caloric intake log keep going?
# 1. What do we do with the food log entries?
#     - Got rid of them for now because it doesn't appear to be worth the effort to wrangle them

# ### Prepare the Environment

# +
from pprint import pprint
import io
from datetime import timedelta
from importlib import reload

import adalib
import acquire
import prepare

import pandas as pd


# -

# **Reload modules to capture changes**

adalib = reload(adalib)
acquire = reload(acquire)
prepare = reload(prepare)

# ## Acquisition <a name="acquisition"></a>


df_cals_in, df_activities, df_food_log = acquire.acquire_fitbit()

# +
# pprint(cals_in)

# +
# pprint(activities)

# +
# pprint(food_log)
# -

# ## Preparation <a name="preparation"></a>

# ### Prepare Caloric Intake Dataframe

# 1. Remove columns from calories_in and convert to int type
# 1. Convert the date column to datetime type

df_cals_in = prepare.df_cals_in_prepare(df_cals_in)

prepare.has_every_day(df_cals_in.date)

df_cals_in.info()

# ### Prepare Activities Dataframe

# 1. Remove commas from calories_burned, steps, minutes_sedentary, activity calories and convert those columns to int type
# 1. Convert date column to datetime type

df_activities = prepare.df_activities_prepare(df_activities)

prepare.has_every_day(df_activities.date)

df_activities.info()

# ### Prepare Food Log Dataframe

# 1. Grab only the data corresponding to the nutritional stats, not the specific food entries, which do not appear to be worth the effort
# 1. Reset the index, rename column to date and convert to datetime type

df_food_log, _ = prepare.df_food_log_prepare(df_food_log)

prepare.has_every_day(df_food_log.date)

df_food_log.info()

# ### Merge DataFrames, Sort, and Index by Date

# 1. Merge the three dataframes on their date column using an outer join
# 1. Drop the columns calories, fat, fiber, carbs, sodium, protein, water because they contain fewer than 10 non-zero entries.

df = prepare.prepare_fitbit(df_cals_in, df_activities, df_food_log)

df.info()

# ### Summarize Data

adalib.summarize(df)

# ### Handle Missing Values

df.isnull().sum()

adalib.df_missing_vals_by_col(df)

df_bad_rows = adalib.df_missing_vals_by_row(df)
df_bad_rows[(df_bad_rows.nmissing > 0) | (df_bad_rows.nempty > 0)]

# **Drop the missing rows**

df = df.dropna()

adalib.df_missing_vals_by_col(df)

# ### Handle Duplicates

# ### Fix Data Types

# ### Handle Outliers

# ### Check Missing Values

# ### Summarize Data

# **Thoughts**
# - What is distance measured in?
# - What does "floors" mean?
# - What is the difference between the "calories in" and "calories" columns?
# -
#
# **Conclusions**
#
# 1. calories_in has a mean of 51; looks like it is mostly 0
# 1. The mean of calories_buned appears to be above the average for a man and way above average for a woman
# 1. Steps and distance metrics appear to match up
# 1. This person is sedentary for on average 13+ hours
# 1. There appear to be days where the Fitbit was not worn or not worn much. min of calories_burned is 799. min of steps and distance and floors is 0. minutes sedentary is 1440 (24 hours).
# 1. columns to drop
#     - calories_in (239 rows are 0)
# 1. After looking at the binned data, it looks like this person was active much of the time by looking at calories_burned and steps.
# 1.

pd.concat([df.head(14), df.tail(14)])

# ## Exploration  <a name="exploration"></a>

# ### Train-Test Split

# ### Visualizations

# +
import matplotlib.pyplot as plt

# %matplotlib inline
import seaborn as sns

plt.figure(figsize=(16, 10))

for i, col in enumerate(
    [
        "calories_in",
        "calories_burned",
        "steps",
        "distance",
        "floors",
        "minutes_sedentary",
        "minutes_lightly_active",
        "minutes_fairly_active",
        "minutes_very_active",
        "activity_calories",
    ]
):
    plot_number = i + 1
    series = df[col]
    plt.subplot(4, 4, plot_number)
    plt.title(col)
    series.hist(bins=20, density=False, cumulative=False, log=False)
# -

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="Blues", annot=True)

# +
plt.figure(figsize=(16, 10))

for i, col in enumerate(
    [
        "calories_in",
        "calories_burned",
        "steps",
        "distance",
        "floors",
        "minutes_sedentary",
        "minutes_lightly_active",
        "minutes_fairly_active",
        "minutes_very_active",
        "activity_calories",
    ]
):
    plot_number = i + 1
    series = df[col]
    plt.subplot(4, 4, plot_number)
    plt.title(col)
    sns.boxplot(data=series)
# -

df["calories_burned_bin"] = pd.qcut(df.calories_burned, q=4)

df.index.min()

sns.swarmplot(
    x="calories_burned_bin", y="activity_calories", data=df, palette="Set2"
)
ax = sns.boxplot(
    x="calories_burned_bin",
    y="activity_calories",
    data=df,
    showcaps=True,
    boxprops={"facecolor": "None"},
    showfliers=True,
    whiskerprops={"linewidth": 0},
)

train = df[:"2018-08"]
test = df["2018-12":]
print(train.nunique())
print(test.nunique())

# ### Statistical Tests

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

# ### Feature Engineering & Selection

# ### Train & Test Models

# ### Summarize Conclusions
