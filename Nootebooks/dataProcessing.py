import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import category_encoders as ce


def processData(data):

    print ("Processing data...")

    data.isna().sum()
    columnName = "booking_status"

    # Zmiana typu danych dla kolumny "booking_status" na numeryczny (0 i 1)

    data[columnName] = data[columnName].map({'Not_Canceled': 0, 'Canceled': 1})

    X_train, X_test, y_train, y_test, num_features, cat_features = splitData(data,columnName)
    X_train_processed, X_test_processed = setTrainngDataSet(X_train, X_test, y_train, num_features, cat_features)

    return X_train_processed, X_test_processed, y_train, y_test


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


def setTrainngDataSet(X_train, X_test, y_train, num_features, cat_features):
    
    # Pipeline dla cech numerycznych (Bez zmian)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline dla cech kategorycznych
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # ### ZMIANA 4: Podmieniamy OneHotEncoder na TargetEncoder
        ('target_encoder', ce.TargetEncoder()) 
    ])

    # Łączenie pipeline'ów (Bez zmian)
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    data_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # ### ZMIANA 5: Target Encoder MUSI widzieć y_train, żeby policzyć prawdopodobieństwa!
    data_pipeline.fit(X_train, y_train)

    features_names = (
        pd.Index(data_pipeline.named_steps['preprocessor'].get_feature_names_out())
        .str.replace('num__', "", regex=False)
        .str.replace('cat__', "", regex=False)
    )

    X_train_processed = pd.DataFrame(data=data_pipeline.transform(X_train), columns = features_names)
    X_test_processed = pd.DataFrame(data=data_pipeline.transform(X_test), columns = features_names)

    print("Processed training dataset: ") 
    print(X_train_processed.head())

    return X_train_processed, X_test_processed




def create_features(df):
    print("Rozpoczynam Inżynierię Cech...")
    data = df.copy()
    #date_str = data['arrival_year'].astype(str) + '-' + \
    #           data['arrival_month'].astype(str) + '-' + \
    #           data['arrival_date'].astype(str)
               
    #full_date = pd.to_datetime(date_str, errors='coerce')
    #data['arrival_day_of_week'] = full_date.dt.dayofweek
    #data['arrival_day_of_week'] = data['arrival_day_of_week'].fillna(-1).astype(int)
    #data['is_arrival_weekend'] = (data['arrival_day_of_week'] >= 5).astype(int)
    #data['total_guests'] = data['no_of_adults'] + data['no_of_children']
    #data['total_nights'] = data['no_of_weekend_nights'] + data['no_of_week_nights']
    data['price_per_person'] = data['avg_price_per_room'] / (data['no_of_adults'] + data['no_of_children']).replace(0, 1)
    # Przedziały: 0-7 dni (Last minute), 8-30 dni (Krótki), 31-90 (Średni), 91-180 (Długi), 180+ (Bardzo długi)
    #bins = [-1, 7, 30, 90, 180, 10000]
    #labels = [0, 1, 2, 3, 4]
    #data['lead_time_category'] = pd.cut(data['lead_time'], bins=bins, labels=labels).astype(int)
    #kolumny_do_usuniecia = [
    #    'arrival_year', 
    #    'arrival_month', 
    #    'arrival_date', 
    #    'avg_price_per_room',
    #    'lead_time_category'
    #]

    #data = data.drop(columns=kolumny_do_usuniecia)
    return data

