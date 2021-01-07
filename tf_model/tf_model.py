import pandas as pd
import numpy as np
from sklearn_model.create_model import load_dataset, attr_label_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

df = load_dataset()

# Select columns for specific transformations and processing
drop_cols = ['Cabin', 'Name', 'PassengerId', 'Ticket']
num_cols = ['Fare', 'Age']
cat_cols = ['Sex', 'Embarked']

X_train, y_train = attr_label_split(df)

# Pipeline to transform numerical attributes. This will replace median of each column with missing values and normalize attributes
num_transfs = [('impute', SimpleImputer(strategy = 'median')), ('std_scaler', Normalizer())]
num_pipeline = Pipeline(num_transfs)

