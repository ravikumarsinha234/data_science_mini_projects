import pandas as pd
from datetime import date
import sys

from sklearn.preprocessing import OrdinalEncoder


def series_report(series, is_ordinal=False, is_continuous=False, is_categorical=False):
    print(f"{series.name}: {series.dtype}")
    if series.isnull().sum() != 0:
        print(
            f"\tMissing in {series.isnull().sum()} rows ({series.isnull().sum()/len(series)*100:.1f}%)"
        )
    if is_ordinal is True:
        print(f"\tRange: {min(series)} - {max(series)}")
    if is_categorical is True:
        for key, value in series.value_counts().items():
            print(f"\t \t{value} : {key}")
    if is_continuous is True:
        print(f"\tMean: {series.mean():.2f}")
        print(f"\tStandard deviation: {series.std():.2f} ")
        print(f"\tMedian: {series.median():.2f}")


# Check command line arguments
if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <input_file>")
    exit(1)

# Read in the data
df = pd.read_csv(sys.argv[1], index_col="employee_id")

# Convert strings to dates for dob and death
df["dob"] = df["dob"].apply(lambda x: date.fromisoformat(x))
df["death"] = df["death"].apply(lambda x: date.fromisoformat(x))

# Show the shape of the dataframe
(row_count, col_count) = df.shape
print(f"*** Basics ***")
print(f"Rows: {row_count:,}")
print(f"Columns: {col_count}")

# Do a report for each column
print(f"\n*** Columns ***")
series_report(df.index, is_ordinal=True)
series_report(df["gender"], is_categorical=True)
series_report(df["height"], is_ordinal=True, is_continuous=True)
series_report(df["waist"], is_ordinal=True, is_continuous=True)
series_report(df["salary"], is_ordinal=True, is_continuous=True)
series_report(df["dob"], is_ordinal=True)
series_report(df["death"], is_ordinal=True)
