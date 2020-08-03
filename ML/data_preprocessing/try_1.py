"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


le = LabelEncoder()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
sc = StandardScaler()


def get_dataset():
    dataset = pd.read_csv('Data.csv')
    features = dataset.iloc[:, :-1].values
    dep_var = dataset.iloc[:, -1].values
    return features, dep_var


def feature_transform(features):
    """
    :param features:
    :return:
    """
    imputer.fit(features[:, 1:3])
    features[:, 1:3] = imputer.transform(features[:, 1:3])
    features = np.array(ct.fit_transform(features))
    return features


def dep_var_encoding(dep_var):
    dep_var = le.fit_transform(dep_var)
    return dep_var


def split_data(features, dep_var):
    return train_test_split(features, dep_var, test_size=0.2, random_state=1)


def feature_scaling(data):
    data[:, 3:] = sc.fit_transform(data[:, 3:])

if __name__ == '__main__':
    """
    """
    features, dep_var = get_dataset()
    features = feature_transform(features)
    dep_var = dep_var_encoding(dep_var)
    features_train, features_test, dep_var_train,\
        dep_var_test = split_data(features, dep_var)

    feature_scaling(features_train)
    feature_scaling(features_test)

    print(features_train)
