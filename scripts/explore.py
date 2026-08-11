import pandas as pd

df = pd.read_csv("data/hotels.csv")

DROP_COLS = [
    "reservation_status", "reservation_status_date",  # leakage
    "company", "agent",                                 # too much missing data
    "country",                                           # high cardinality
    "assigned_room_type",                                # not known at booking time
    "arrival_date_day_of_month",                         # low signal, high noise
]

df = df.drop(columns=DROP_COLS)

TARGET = "is_canceled"
FEATURES = [c for c in df.columns if c != TARGET]

CATEGORICAL = df[FEATURES].select_dtypes(include="object").columns.tolist()
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

print(f"Total features: {len(FEATURES)}")
print(f"\nCategorical ({len(CATEGORICAL)}):", CATEGORICAL)
print(f"\nNumeric ({len(NUMERIC)}):", NUMERIC)
