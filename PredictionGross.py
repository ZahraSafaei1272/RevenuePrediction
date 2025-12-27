import pandas as pd
import numpy as np
import ast
import sqlite3
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb

# -------------------------------------------------
# Data Loading
# -------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw movie metadata CSV."""
    return pd.read_csv(filepath, low_memory=False)


# -------------------------------------------------
# Column Cleaning
# -------------------------------------------------

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not used in modeling."""
    cols_to_drop = [
        'adult', 'overview', 'belongs_to_collection', 'homepage',
        'original_language', 'spoken_languages', 'status',
        'tagline', 'title', 'video', 'poster_path',
        'production_companies'
    ]
    return df.drop(columns=cols_to_drop, errors='ignore')


# -------------------------------------------------
# Revenue & Budget Processing
# -------------------------------------------------

def clean_revenue_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Filter invalid revenues, impute budgets, and log-transform."""
    df = df[df['revenue'].fillna(0) != 0].reset_index(drop=True)

    df['revenue'] = df['revenue'].astype(float)
    df['budget'] = df['budget'].astype(float).replace(0, np.nan)

    imputer = KNNImputer(n_neighbors=5)
    df['budget'] = imputer.fit_transform(df[['budget']]).round()

    df['log_revenue'] = np.log1p(df['revenue'])
    df['log_budget'] = np.log1p(df['budget'])

    return df


# -------------------------------------------------
# Genre Processing
# -------------------------------------------------

def parse_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Convert genre JSON strings into dummy variables."""
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['genres'] = df['genres'].apply(
        lambda x: [d['name'] for d in x] if isinstance(x, list) else []
    )

    # Manual corrections (dataset-specific and fragile)
    genre_fixes = {
        6386: ['Comedy'],
        6937: ['Action', 'Drama', 'Romance'],
        7049: ['Thriller'],
        7085: ['Drama'],
        7090: ['Adventure'],
        7271: ['Romance', 'Drama'],
        7304: ['Comedy'],
        7347: ['Drama'],
        7348: ['Action', 'Crime'],
        7359: ['Comedy', 'Crime', 'Mystery'],
        6401: ['Drama', 'Romance'],
        6699: ['Drama', 'Fantasy', 'Mystery'],
        6726: ['Drama'],
        6757: ['Adventure', 'Biography', 'Drama', 'Romance'],
        2085: ['Comedy', 'Drama'],
        2306: ['Comedy'],
        3134: ['Documentary', 'Drama', 'War'],
        3481: ['Musical', 'Comedy'],
        4424: ['Documentary', 'Biography', 'Family'],
        4932: ['Comedy', 'Crime', 'Drama'],
        5009: ['Action', 'Drama'],
        6193: ['Action', 'Crime', 'Thriller'],
        6371: ['Action', 'Crime', 'Drama', 'Thriller']
    }

    for idx, genres in genre_fixes.items():
        if idx in df.index:
            df.at[idx, 'genres'] = genres

    genre_dummies = df['genres'].apply(lambda x: ','.join(x)).str.get_dummies(',')
    return df.join(genre_dummies)


# -------------------------------------------------
# Runtime Fixes
# -------------------------------------------------

def fix_runtime(df: pd.DataFrame) -> pd.DataFrame:
    """Manually fix invalid or missing runtimes."""
    runtime_fixes = {
        6371: 91, 6443: 96, 6454: 96, 6522: 140, 6531: 84,
        6544: 86, 6545: 93, 6612: 86, 6638: 83, 6713: 111,
        6727: 103, 6749: 92, 6752: 104, 6850: 140, 7237: 86,
        7337: 86, 7352: 108, 7354: 90, 7359: 86, 7394: 98,
        7403: 93, 305: 86, 6216: 91, 6765: 100, 7022: 93,
        7085: 90, 7297: 130
    }

    for idx, value in runtime_fixes.items():
        if idx in df.index:
            df.at[idx, 'runtime'] = value

    return df


# -------------------------------------------------
# Date Features
# -------------------------------------------------

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from release date."""
    df.loc[1631, 'release_date'] = '2000-05-10'
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    df['release_year'] = df['release_date'].dt.year
    df['movie_age'] = pd.Timestamp.now().year - df['release_year']
    df['release_quarter'] = df['release_date'].dt.quarter

    return df


# -------------------------------------------------
# Database Persistence
# -------------------------------------------------

def save_to_sqlite(df: pd.DataFrame, db_name: str, table_name: str):
    """Persist preprocessed data into SQLite."""
    conn = sqlite3.connect(db_name)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()


# -------------------------------------------------
# Modeling
# -------------------------------------------------

def train_xgboost(X, y):
    """Train XGBoost regressor."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=123
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=8,
        gamma=0.5,
        min_child_weight=10,
        subsample=0.6,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=10,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds_train = model.predict(X_train)
    preds = model.predict(X_test)

    return model, r2_score(y_train, preds_train), r2_score(y_test, preds)


# -------------------------------------------------
# Pipeline Execution
# -------------------------------------------------

def main():
    df = load_data('movies_metadata.csv')
    df = drop_unused_columns(df)
    df = clean_revenue_budget(df)
    df = parse_genres(df)
    df = fix_runtime(df)
    df = create_date_features(df)


    feature_cols = [
        'runtime', 'vote_average', 'vote_count', 'log_budget',
        'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
        'Foreign', 'History', 'Horror', 'Music', 'Musical',
        'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
        'Thriller', 'War', 'Western',
        'release_year', 'movie_age', 'release_quarter'
    ]
    save_to_sqlite(df[['id']+feature_cols], 'movies_database.db', 'movies_preprocessed')

    X = df[feature_cols]
    y = df['log_revenue']

    model, r2_train, r2_test = train_xgboost(X, y)
    print(f'R² Score Train: {r2_train:.4f}')
    print(f"R² Score Test: {r2_test:.4f}")


if __name__ == "__main__":
    main()




