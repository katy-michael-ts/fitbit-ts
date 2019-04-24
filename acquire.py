from enum import Enum, auto
import io
import os


import pandas as pd

DATADIR = "fitbit"


CALS_IN_HEADER = "Date,Calories In"
ACTIVITIES_HEADER = (
    "Date,Calories Burned,Steps,Distance,Floors,"
    "Minutes Sedentary,Minutes Lightly Active,Minutes Fairly Active,"
    "Minutes Very Active,Activity Calories"
)
FOOD_LOG_HEADER = "Daily Totals"


class CSVType(Enum):
    CALS_IN = auto()
    ACTIVITIES = auto()
    FOOD_LOG = auto()


def read_data() -> tuple:
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

    return cals_in, activities, food_log


def acquire_fitbit() -> tuple:
    cals_in, activities, food_log = read_data()

    cals_in_cols = ["date", "calories_in"]
    df_cals_in = pd.read_csv(
        io.StringIO("".join(cals_in)), header=None, names=cals_in_cols
    )

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

    # food_log_cols = ["date", "calories", "fat", "fiber", "carbs", "sodium",
    #                  "protein", "water"]
    food_log_cols = ["date", "column", "value"]
    df_food_log = pd.read_csv(
        io.StringIO("".join(food_log)), header=None, names=food_log_cols
    )

    return df_cals_in, df_activities, df_food_log
