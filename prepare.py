from datetime import timedelta

import pandas as pd


def df_col_to_datetime(df: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
    datetimed = pd.to_datetime(df[col], **kwargs)
    out = df.copy()
    out[col] = datetimed
    return out


def series_remove_commas(series: pd.Series) -> pd.Series:
    return series.str.replace(",", "")


def has_every_day(date_series: pd.Series) -> bool:
    return timedelta(date_series.nunique()) == (
        date_series.max() - date_series.min()
    ) + timedelta(days=1)


def df_cals_in_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df["calories_in"] = series_remove_commas(df.calories_in)
    df = df_col_to_datetime(df, "date", format="%Y-%m-%d")
    return df.astype({"calories_in": int})


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


# I need to iterate over the DataFrame, use the date as the index, make
# columns for calories, fat, etc and put the value there
def df_food_log_prepare(df: pd.DataFrame) -> tuple:
    columns = (
        "calories",
        "fat",
        "fiber",
        "carbs",
        "sodium",
        "protein",
        "water",
    )
    out_of_place = []
    out = pd.DataFrame()
    for index, row in df.iterrows():
        if isinstance(row["column"], str):
            col_lower = row["column"].lower()
            if col_lower in columns:
                out.loc[row["date"], col_lower] = row["value"]
            else:
                out_of_place.append(row)
        else:
            out_of_place.append(row)

    out = out.reset_index()
    out = out.rename(columns={"index": "date"})

    return df_col_to_datetime(out, "date", format="%Y-%m-%d"), out_of_place


def prepare_fitbit(
    df_cals_in: pd.DataFrame,
    df_activities: pd.DataFrame,
    df_food_log: pd.DataFrame,
) -> pd.DataFrame:

    df = df_cals_in.merge(df_activities, how="outer", on="date")
    df = df.merge(df_food_log, how="outer", on="date")

    df = df.drop(
        columns=[
            "calories_in",
            "calories",
            "fat",
            "fiber",
            "carbs",
            "sodium",
            "protein",
            "water",
        ]
    )

    return df.sort_values("date").set_index("date")
