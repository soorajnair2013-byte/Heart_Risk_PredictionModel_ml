# import Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE


def preprocessing(df):

    # Step 1: Remove duplicate rows
    df = df.drop_duplicates()

    # Step 2: Encode target column
    df["class"] = df["class"].map({"negative": 0, "positive": 1})

    # Step 3: Split features and target
    X = df.drop(columns="class")
    y = df["class"]

    # Step 4: Identify categorical and numerical columns
    categorical_data = X.select_dtypes(include="object").columns
    numerical_data = X.select_dtypes(exclude="object").columns

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1,
        stratify=y
    )

    # Step 6: Numerical pipeline
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    # Step 7: Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    # Step 8: Column Transformer
    transformer = ColumnTransformer([
        ("num", numerical_pipeline, numerical_data),
        ("cat", categorical_pipeline, categorical_data)
    ])

    # Step 9: Transform training and testing data
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)

    # Step 10: Apply SMOTE on training data only
    sm = SMOTE(random_state=1)
    X_train, y_train = sm.fit_resample(X_train, y_train) # type: ignore

    return X_train, X_test, y_train, y_test, transformer