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
import os
from pprint import pprint
import io
from datetime import timedelta
from importlib import reload

import adalib
import acquire
import prepare

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
import pandas as pd
import math
from statsmodels.tsa.api import Holt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric, plot_forecast_component

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

# ## Modeling <a name="modeling"></a>

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

# +
d_df = df

d_df['y'] = d_df.calories_burned
d_df['ds'] = d_df.index
d_df = d_df.groupby(['ds'])['y'].sum().reset_index()
# -

d_df.head()

plt.figure(figsize=(16,6))
sns.lineplot(d_df.ds, d_df.y)

# +
d_df['cap'] = 5000
d_df['floor'] = 2100

m = Prophet(daily_seasonality=True, growth='logistic', changepoint_range=0.9)
m.fit(d_df)
# -

future = m.make_future_dataframe(periods=8)
future['cap'] = 5000
future['floor'] = 2100
print(future.head())
print(future.tail())
print(d_df.tail())

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)

# +
# cross_validation(m, initial = 730, period = 180, horizon = 365, units = 'days')
df_cv = cross_validation(m, horizon='30 days')

df_p = performance_metrics(df_cv)
df_p.head(5)
# -

fig3 = plot_cross_validation_metric(df_cv, metric='rmse')

# +
aggregation = 'sum'

train = df[:'2018-09'].activity_calories.resample('W').agg(aggregation)
test = df['2018-10':].activity_calories.resample('W').agg(aggregation)
# -

plt.figure(figsize=(10, 6))
plt.plot(train)
plt.plot(test)
plt.show()

test















# ### Summarize Conclusions
