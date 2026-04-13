from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# import category_encoders as ce

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score


def processData(data):

    print ("Processing data...")

    data.isna().sum()
    columnName = "booking_status"

    # Zmiana typu danych dla kolumny "booking_status" na numeryczny (0 i 1)

    data[columnName] = data[columnName].map({'Not_Canceled': 0, 'Canceled': 1})

    X_train, X_test, y_train, y_test, num_features, cat_features = splitData(data,columnName)

    # Dodanie zbioru walidacyjnego z danych treningowych
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=2/9, random_state=123, stratify=y_train)

    data_pipeline = setPipeline(X_train, y_train, num_features, cat_features)

    all_names = (
        pd.Index(data_pipeline.named_steps['preprocessor'].get_feature_names_out())
        .str.replace('num__', "", regex=False)
        .str.replace('cat__', "", regex=False)
    )

    ## Wybranie tych cech, które zostały wybrane przez Feature Selector
    features_names = all_names[data_pipeline.named_steps['selector'].get_support()]

    X_train_processed = pd.DataFrame(data=data_pipeline.transform(X_train), columns = features_names)
    X_test_processed = pd.DataFrame(data=data_pipeline.transform(X_test), columns = features_names)
    X_val_processed = pd.DataFrame(data=data_pipeline.transform(X_val), columns = features_names)

    print ("Features selected by Feature Selector: ", features_names.tolist())


    return X_train_processed, X_test_processed, X_val_processed, y_train, y_test, y_val


def splitData(data, columnName):
    
    print("Splitting data...")

    # Rozdzielenie zbiorów pod kątem statusu rezerwacji

    X = data.drop(columnName, axis=1)
    y = data[columnName]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123, stratify=y)

    # Podział na cechy numeryczne i kategoryczne

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print ("All feratures: ", X_train.columns.tolist())
    print ("Numerical features: ", num_features)
    print ("Categorical features: ", cat_features)

    return X_train, X_test, y_train, y_test, num_features, cat_features


def setPipeline(X_train, y_train, num_features, cat_features):
    
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
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_classif, k=15))
    ])

    # Dostawienie przetworzonych danych do danych treningowych

    data_pipeline.fit(X_train, y_train)

    return data_pipeline

def winsorizeData(data, columnsToWinsorize):

    for column in columnsToWinsorize:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower=lowerBound, upper=upperBound)

def startFeatureEngineering(data):
    # Dropnięcie niepotrzebnych kolumn, które nie są istotne dla modelu lub mogą wprowadzać szum
    data = data.drop(columns=['Booking_ID', 'no_of_previous_bookings_not_canceled', 'arrival_date', 'arrival_year'])

    # Grupowanie rzadkich kategorii w kolumnie 'room_type_reserved' do jednej kategorii 'Other'
    room_types = ['Room_Type_1', 'Room_Type_4']
    data['room_type_reserved'] = np.where(data['room_type_reserved'].isin(room_types), data['room_type_reserved'], 'Other')

    # Grupowanie no_of_previous_cancellations do kategorii na has_no_previous_cancellations (0) i has_previous_cancellations (1)
    data['has_previous_cancellations'] = np.where(data['no_of_previous_cancellations'] > 0, 1, 0)
    data = data.drop(columns=['no_of_previous_cancellations'])

    # Gupowanie number_of_children do kategorii 0, 1, 2, >2
    data['no_of_children'] = np.where(data['no_of_children'] > 2, '>2', data['no_of_children'].astype(str))

    # Grupowanie no_of_weekend_nights do kategorii 0, 1, 2, >2
    data['no_of_weekend_nights'] = np.where(data['no_of_weekend_nights'] > 2, '>2', data['no_of_weekend_nights'].astype(str))

    # Grupowanie no_of_week_nights do kategorii 0, 1, 2, 3, 4, 5, 6, 7, >7
    data['no_of_week_nights'] = np.where(data['no_of_week_nights'] > 7, '>7', data['no_of_week_nights'].astype(str))

    # Czyszczenie z outlierów zmiennych numerycznych ciągłych za pomocą winsoryzacji
    columnsToWinsorize = ['lead_time', 'avg_price_per_room']
    winsorizeData(data, columnsToWinsorize)

    # Łączenie lead_time i avg_price_per_room w jedną cechę price_lead_time_ratio
    data['price_lead_time_ratio'] = data['avg_price_per_room'] * (data['lead_time'])

    return data


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

