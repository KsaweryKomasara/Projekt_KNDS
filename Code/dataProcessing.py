import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score



def processData(data):
    print ("Processing data...")
    data.isna().sum()
    columnName = "booking_status"
    X_train, X_test, y_train, y_test, num_features, cat_features = splitData(data,columnName)
    setTrainngDataSet(X_train, X_test, num_features, cat_features)


def splitData(data, columnName):
    print("Splitting data...")

    # Rozdzielenie zbiorów pod kątem statusu rezerwacji

    X = data.drop(columnName, axis=1)
    y = data[columnName]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Podział na cechy numeryczne i kategoryczne

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print ("All feratures: ", X_train.columns.tolist())
    print ("Numerical features: ", num_features)
    print ("Categorical features: ", cat_features)

    return X_train, X_test, y_train, y_test, num_features, cat_features


def setTrainngDataSet(X_train, X_test, num_features, cat_features):
    
    # Pipeline dla cech numerycznych

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline dla cech kategorycznych

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Łączenie pipeline'ów

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    data_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # Dostawienie przetworzonych danych do danych treningowych

    data_pipeline.fit(X_train)

    features_names = (
        pd.Index(data_pipeline.named_steps['preprocessor'].get_feature_names_out())
        .str.replace('num__', '', regex=False)
        .str.replace('cat__', '', regex=False)
    )

    X_train_processed = pd.DataFrame(data=data_pipeline.transform(X_train).toarray(), columns = features_names)
    X_test_processed = pd.DataFrame(data=data_pipeline.transform(X_test).toarray(), columns = features_names)

    print("Processed training dataset: ", X_train_processed.head)

