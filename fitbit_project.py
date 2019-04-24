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
# 1. The wearer appears to be fairly active, but not mostly mobile activity, possibly stationary like lifting weights.
# 1. Probably not someone in a drug trial because there are not many entries in the food log. We would expect them to be logging food to see if there are any interactions.
# 1. The wearer is likely not wearing the tracker while sleeping. Average inactivity/activity minutes is 16-17 hours per day.
#     - Wouldn't expect a person testing fitness equipment to wear it that long
#     - Would make sense for an employee or drug trial participant (but we would expect the drug trial participant to wear it while sleeping)
# 1. Likely not a person testing fitness equipment because there are food log entries for one week; and also likely not a drug trial participant for the same reason
#     - Makes it more likely to be an employee who lost the motivation to log food
# - Next two week
# 1. Looks like person stopped wearing tracker on 12/7/18 because there are food log entries and caloric intake entries for dates after the 7th, but the activity log stops on the 6th.

# ### Thoughts & Questions
#
# 1. What does the weekend data look like? This may tell us whether they work there or are in a drug trial  (likely to wear the fitbit on the weekend) or are testing fitness equipment (not likely to wear on weekend)

# ### Prepare the Environment

import os
from pprint import pprint
from enum import Enum, auto
import io
from datetime import timedelta
from pylab import rcParams
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
import adalib
import pandas as pd
import math
from statsmodels.tsa.api import Holt
from fbprophet import Prophet

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
    with open(DATADIR + "/" + file) as fp:
        lines = fp.readlines()
        csvtype = None
        date = ""
        for line in lines:
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
            elif (
                line.startswith(CALS_IN_HEADER)
                or line.startswith(ACTIVITIES_HEADER)
                or line.startswith(FOOD_LOG_HEADER)
            ):
                continue

            # IT'S DATA!
            elif csvtype == CSVType.CALS_IN:
                cals_in.append(line)
            elif csvtype == CSVType.ACTIVITIES:
                activities.append(line)
            elif csvtype == CSVType.FOOD_LOG:
                if line.startswith('""'):
                    line = line.replace(
                        '""', f'"{date[:4]}-{date[4:6]}-{date[6:]}"'
                    )
                food_log.append(line)


# +
# pprint(cals_in)

# +
# pprint(activities)

# +
# pprint(food_log)
# -

# ## Preparation <a name="preparation"></a>

# ### Prepare Caloric Intake Dataframe

cals_in_cols = ["date", "calories_in"]
df_cals_in = pd.read_csv(
    io.StringIO("".join(cals_in)), header=None, names=cals_in_cols
)


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


def has_every_day(date_series: pd.Series) -> bool:
    return timedelta(date_series.nunique()) == (
        date_series.max() - date_series.min()
    ) + timedelta(days=1)


has_every_day(df_cals_in.date)

df_cals_in.info()

# ### Prepare Activities Dataframe

activities_cols = [
    "date",
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
df_activities = pd.read_csv(
    io.StringIO("".join(activities)), header=None, names=activities_cols
)


def df_activities_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df_col_to_datetime(df, "date", format="%Y-%m-%d")

    df["calories_burned"] = series_remove_commas(df.calories_burned)
    df["steps"] = series_remove_commas(df.steps)
    df["minutes_sedentary"] = series_remove_commas(df.minutes_sedentary)
    df["activity_calories"] = series_remove_commas(df.activity_calories)

    return df.astype(
        {
            "calories_burned": int,
            "steps": int,
            "minutes_sedentary": int,
            "activity_calories": int,
        }
    )


df_activities = df_activities_prepare(df_activities)

has_every_day(df_activities.date)

df_activities.info()

# ### Prepare Food Log Dataframe

# food_log_cols = ["date", "calories", "fat", "fiber", "carbs", "sodium",
#                  "protein", "water"]
food_log_cols = ["date", "column", "value"]
df_food_log = pd.read_csv(
    io.StringIO("".join(food_log)), header=None, names=food_log_cols
)

df_food_log.info()

# **I need to iterate over the DataFrame, use the date as the index, make columns for calories, fat, etc and put the value there**

# +
columns = ("calories", "fat", "fiber", "carbs", "sodium", "protein", "water")
out_of_place = []
out = pd.DataFrame()
for index, row in df_food_log.iterrows():
    if isinstance(row["column"], str):
        col_lower = row["column"].lower()
        if col_lower in columns:
            out.loc[row["date"], col_lower] = row["value"]
        else:
            out_of_place.append(row)
    else:
        out_of_place.append(row)

#     print(row["column"], row["value"])
# -

out = out.reset_index()
out = out.rename(columns={"index": "date"})


# +
def df_food_log_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df_col_to_datetime(df, "date", format="%Y-%m-%d")
    return df


#     df["calories_burned"] = series_remove_commas(df.calories_burned)
#     df["steps"] = series_remove_commas(df.steps)
#     df["minutes_sedentary"] = series_remove_commas(df.minutes_sedentary)
#     df["activity_calories"] = series_remove_commas(df.activity_calories)

#     return df.astype({"calories_burned": int, "steps": int, "minutes_sedentary": int, "activity_calories": int})


# -

out = df_food_log_prepare(out)

out.info()

# ### Start Merging Dataframes

df = df_cals_in.merge(df_activities, how="outer", on="date")

df.info()

df = df.merge(out, how="outer", on="date")

df = df.sort_values("date").set_index("date")

# ### What to do with the extras? <a name="todo-extra-data"></a>

print(out_of_place[0])
print()
print(out_of_place[1])
print()
print(out_of_place[2])

df.describe()

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

#Weekly
calories = train.resample('W').calories_burned.mean()

calories.head()

calories.plot()

#Monthly
calories.resample('MS').mean().plot()

#5 period rolling mean and plot
calories.rolling(5).mean().plot(figsize=(12, 4))

#10 period difference and plot
calories.diff(periods=10).plot(figsize=(12, 4))

#lag plot
pd.plotting.lag_plot(calories)

#pearson correlation
df_corr = pd.concat([calories.shift(1), calories], axis=1)
df_corr.columns = ['t-1','t+1']
result = df_corr.corr()
print(result)

#autocorrelation plot
pd.plotting.autocorrelation_plot(calories)

#partial autocorrelation plot
sm.graphics.tsa.plot_pacf(calories)

# ### Modeling

# +
# df = df.dropna

# +
aggregation = 'sum'

train = df[:'2018-09'].calories_burned.resample('D').agg(aggregation)
test = df['2018-10':].calories_burned.resample('D').agg(aggregation)
# -

print('Observations: %d' % (len(train.values) + len(test.values)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

pd.concat([train.head(3), train.tail(3)])

plt.figure(figsize=(10, 6))
plt.plot(train)
plt.plot(test)
plt.show()

# ### SIMPLE AVERAGE

yhat = pd.DataFrame(dict(actual=test))

yhat['avg_forecast'] = train.mean()
yhat.head()


# +
def plot_data_and_predictions(predictions, label):
    plt.figure(figsize=(10, 8))

    plt.plot(train,label='Train')
    plt.plot(test, label='Test')
    plt.plot(predictions, label=label, linewidth=5)

    plt.legend(loc='best')
    plt.show()


def evaluate(actual, predictions, output=True):
    mse = metrics.mean_squared_error(actual, predictions)
    rmse = math.sqrt(mse)

    if output:
        print('MSE:  {}'.format(mse))
        print('RMSE: {}'.format(rmse))
    else:
        return mse, rmse    

def plot_and_eval(predictions, actual=test, metric_fmt='{:.2f}', linewidth=4):
    if type(predictions) is not list:
        predictions = [predictions]

    plt.figure(figsize=(16, 8))
    plt.plot(train,label='Train')
    plt.plot(test, label='Test')

    for yhat in predictions:
        mse, rmse = evaluate(actual, yhat, output=False)        
        label = f'{yhat.name}'
        if len(predictions) > 1:
            label = f'{label} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse)
        plt.plot(yhat, label=label, linewidth=linewidth)

    if len(predictions) == 1:
        label = f'{label} -- MSE: {metric_fmt} RMSE: {metric_fmt}'.format(mse, rmse)
        plt.title(label)

    plt.legend(loc='best')
    plt.show() 


# -

plot_and_eval(yhat.avg_forecast)

# ### MOVING AVERAGE

periods = 7
yhat['moving_avg_forecast_7'] = train.rolling(7).mean().iloc[-1] 

plot_and_eval(yhat.moving_avg_forecast_7)

# +
period_vals = [7, 20, 30, 60, 90]

for periods in period_vals:
    yhat[f'moving_avg_forecast_{periods}'] = train.rolling(periods).mean().iloc[-1] 

forecasts = [yhat[f'moving_avg_forecast_{p}'] for p in period_vals]

plot_and_eval(forecasts, linewidth=2)
# -

# ## Holts Linear Trend

sm.tsa.seasonal_decompose(train).plot()
result = sm.tsa.stattools.adfuller(train)
plt.show()

# +
holt = Holt(train).fit(smoothing_level=.2, smoothing_slope=.1)

yhat['holt_linear'] = holt.forecast(test.shape[0])
# -

plot_and_eval(yhat.holt_linear)

# ## Prophet

df['y'] = df.calories_burned
df['ds'] = df.Date
df = df.groupby(['ds'])['y'].sum().reset_index()



# ### Summarize Data

# ### Handle Missing Values

df.isnull().sum()

df[df.isnull().any(axis=1)]

# **Impute calories_burned using the mean**

# **Impute steps by mean**

# **Impute distance by mean**

# **Impute floors with mean**

#

#

# ### Handle Duplicates

# ### Fix Data Types

# ### Handle Outliers

# ### Check Missing Values

# ### Summarize Data

df.info()

adalib.summarize(df)

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

# ### Statistical Tests

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

# ### Feature Engineering & Selection

# ### Train & Test Models

# ### Summarize Conclusions
