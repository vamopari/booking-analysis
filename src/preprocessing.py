import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

TARGET = "is_canceled"

DROP_COLS = [
    "reservation_status", "reservation_status_date",
    "company", "agent", "country",
    "assigned_room_type", "arrival_date_day_of_month",
]

CATEGORICAL = [
    "hotel", "arrival_date_month", "meal", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type",
    "customer_type",
]

NUMERIC = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults",
    "children", "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests",
]

FEATURES = CATEGORICAL + NUMERIC


def load_clean_data(path="data/hotels.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS)
    return df


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


if __name__ == "__main__":
    df = load_clean_data()
    X = df[FEATURES]
    y = df[TARGET]

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    print("Original X shape:", X.shape)
    print("Transformed X shape:", X_transformed.shape)
    print("\nFirst transformed row (first 10 values):")
    print(X_transformed[0][:10])


def split_data(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nTrain target balance:")
    print(y_train.value_counts(normalize=True))
    print("\nTest target balance:")
    print(y_test.value_counts(normalize=True))

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)   # fit on TRAIN only
    X_test_transformed = preprocessor.transform(X_test)          # just apply to TEST

    print("\nTransformed train shape:", X_train_transformed.shape)
    print("Transformed test shape:", X_test_transformed.shape)
