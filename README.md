# Predict Survival on Titanic  

Getting, analyzing, preprocessing the Titanic dataset which can be found [here](https://www.kaggle.com/c/titanic/data).

## How-to get the dataset  

- Install Kaggle Python package

```
pip install kaggle
```

- Download dataset using Kaggle API

```
kaggle competitions download -c titanic
```

## Models

Here are the models trained to the dataset and their Root Mean Squared Errors

| Model                                  | RMSE               |
|----------------------------------------|--------------------|
| Support Vector Machine Classifier(SVC) | 0.3955054753168236 |
| Random Forest Classifier               | 0.454647017602145  |
| Gaussian Naive Bayes                   | 0.4901259671626783 |
