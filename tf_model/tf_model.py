import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import tensorflow as tf

def load_dataset(path = '../titanic'):
  dataset_path = os.path.join(path, 'train.csv')
  return pd.read_csv(dataset_path)

df = load_dataset()

# Select columns for specific transformations and processing
drop_cols = ['Cabin', 'Name', 'PassengerId', 'Ticket']
num_cols = ['Fare', 'Age']
cat_cols = ['Sex', 'Embarked']

def attr_label_split(df):
  y = df['Survived']
  X = df.drop('Survived', axis = 1)
  return X, y

X_train, y_train = attr_label_split(df)

# Pipeline to transform numerical attributes. This will replace median of each column with missing values and normalize attributes
num_transfs = [('impute', SimpleImputer(strategy = 'median')), ('normalizer', Normalizer())]
num_pipeline = Pipeline(num_transfs)

# Pipeline to transform categorical attributes. This will replace most frequent of each column with missing values and assign numbers for each categories.
cat_transfs = [('impute', SimpleImputer(strategy = 'most_frequent')), ('encoder', OneHotEncoder())]
cat_pipeline = Pipeline(cat_transfs)

# The complete pipeline to transform entire dataframes
all_transfs = [('numeric', num_pipeline, num_cols), ('categorical', cat_pipeline, cat_cols), ('drops', 'drop', drop_cols)]
full_pipeline = ColumnTransformer(all_transfs, remainder = 'passthrough')

X_train_transformed = full_pipeline.fit_transform(X_train)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(5, activation = 'relu'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])

model.fit(X_train_transformed, y_train, epochs = 25, verbose = 2)

model.save('titanic_model')