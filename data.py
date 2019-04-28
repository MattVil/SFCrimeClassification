import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from utils import PROJECT_PATH, DATA_PATH

def load_data():
    """"""
    train_data = pd.read_csv(DATA_PATH + "/train.csv", parse_dates = ['Dates'])
    test_data = pd.read_csv(DATA_PATH + "/test.csv", parse_dates = ['Dates'])
    return train_data, test_data

def data_preprocessing(train_data, test_data):
    """"""
    le = preprocessing.LabelEncoder()
    crime = le.fit_transform(train_data.Category)
    days = pd.get_dummies(train_data.DayOfWeek)
    district = pd.get_dummies(train_data.PdDistrict)
    year = train_data.Dates.dt.year
    hour = train_data.Dates.dt.hour
    day = train_data.Dates.dt.day
    x = train_data.X
    y = train_data.Y
    training = pd.concat([year, day, hour, days, district, x, y], axis=1)
    training['crime'] = crime

    days = pd.get_dummies(test_data.DayOfWeek)
    district = pd.get_dummies(test_data.PdDistrict)
    year = test_data.Dates.dt.year
    hour = test_data.Dates.dt.hour
    day = test_data.Dates.dt.day
    x = test_data.X
    y = test_data.Y
    testing = pd.concat([year, day, hour, days, district, x, y], axis=1)

    x_train = training.iloc[:,:-1]
    y_train = training['crime']

    # x_train, x_valid, y_train, y_valid = train_test_split(x_train,
    #                                                       y_train,
    #                                                       test_size=0.33,
    #                                                       random_state=42)

    return x_train.values, y_train.values, testing.values




if __name__ == '__main__':
    train_data, test_data = load_data()
    x_train, y_train, x_test = data_preprocessing(train_data, test_data)

    print(x_train.head())
    print(y_train.head())
    print(x_valid.head())
    print(y_valid.head())
    print(x_test.head())
