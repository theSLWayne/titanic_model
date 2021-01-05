import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC

# Function to load the dataset
def load_dataset(path = 'titanic'):
  dataset_path = os.path.join(path, 'train.csv')
  return pd.read_csv(dataset_path)

# Load dataset
df = load_dataset()

# Select columns for specific transformations and processing
drop_cols = ['Cabin', 'Name', 'PassengerId', 'Ticket']
num_cols = ['Fare', 'Age']
cat_cols = ['Sex', 'Embarked']

# Separate attributes and labels
def attr_label_split(df):
  y = df['Survived']
  X = df.drop('Survived', axis = 1)
  return X, y

X_train, y_train = attr_label_split(df)

# Pipeline to transform numerical attributes. This will replace median of each column with missing values and standardize attributes
num_transfs = [('impute', SimpleImputer(strategy = 'median')), ('std_scaler', StandardScaler())]
num_pipeline = Pipeline(num_transfs)

# Pipeline to transform categorical attributes. This will replace most frequent of each column with missing values and assign numbers for each categories.
cat_transfs = [('impute', SimpleImputer(strategy = 'most_frequent')), ('encoder', OrdinalEncoder())]
cat_pipeline = Pipeline(cat_transfs)

# The complete pipeline to transform entire dataframes
all_transfs = [('numeric', num_pipeline, num_cols), ('categorical', cat_pipeline, cat_cols), ('drops', 'drop', drop_cols)]
full_pipeline = ColumnTransformer(all_transfs, remainder = 'passthrough')

# Transform the dataset
X_train_transformed = full_pipeline.fit_transform(X_train)

# Train a SVM Classifier
svm_model = SVC(C = 1000, gamma = 'scale', kernel = 'rbf')

svm_model.fit(X_train_transformed, y_train)

# Save the model 
pickle.dump(svm_model, open('model.pkl', 'wb'))

# Save the pipeline
pickle.dump(full_pipeline, open('pipeline.pkl', 'wb'))