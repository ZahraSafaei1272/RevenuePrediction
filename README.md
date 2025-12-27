## Movie Revenue Prediction with XGBoost

This project preprocesses movie metadata and trains an XGBoost regression model to predict movie revenue using financial, temporal, and genre-based features.

# Overview

**The pipeline:**

- Cleans and filters raw movie metadata

- Imputes missing budgets using KNN

- Applies log transformations to revenue and budget

- Parses and encodes movie genres

- Engineers time-based features from release dates

- Stores preprocessed data in SQLite

- Trains and evaluates an XGBoost regression model

## Input Data

This project depends on `TMDb (The Movie Database)` which must be downloaded manually.

Download from:

[tmdb-movies-dataset-2023-930k-movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)

Required files:

- `movies_metadata.csv`


Raw movie metadata containing budget, revenue, genres, runtime, votes, and release dates.

# Main Steps

1. **Data Cleaning**

- Drops unused columns

- Removes movies with zero revenue

- Imputes missing budgets

2. **Feature Engineering**

- One-hot encoding of genres

- Runtime and release date corrections

- Creation of year, age, and quarter features

3. **Modeling**

- XGBoost regression

- Train/test split

- R² evaluation on train and test sets

## Output

- SQLite database: `movies_database.db`

- Console output:

   Train and test R² scores

## How to Run
```bash
python PredictionGross.py
```

## Dependencies

- pandas

- numpy

- scikit-learn

- xgboost

- sqlite3 (standard library)
