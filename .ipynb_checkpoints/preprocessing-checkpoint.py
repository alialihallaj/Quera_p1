import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import (
    PowerTransformer, RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
)
from sklearn.base import BaseEstimator, TransformerMixin

def preprocessing(df, cc):
    #general preprocessing
    df.drop_duplicates()

    df['created_at_month'] = pd.to_datetime(df['created_at_month']).dt.month

    df[df['land_size'] < df['building_size'] &]

    df.drop(['description', 'title', 'rent_to_single', 'city_slug', 'neighborhood_slug', 'rent_mode', 'credit_mode', 'price_mode'], axis=1, inplace=True)

    def neighborhood(area_csv):
        from sklearn.cluster import KMeans

        df['cluster'] = None

        area_df = pd.read_csv(area_csv)
    
        for city, area in area_df:
            n_points = len(df[df['city_slug'] == city])
                        
            n_clusters = int(min(300, max(5, (n_points / 2000) + np.sqrt(area))))
            
            print(f"Clustering {city}: {n_points} properties → {n_clusters} clusters")
            
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(group[['lat', 'long']])
            
            df.loc[group.index, 'cluster'] = [f"{city}_{c}" for c in labels]
    



    
def handling_missing_values(df):
    #below function detect integer columns so for nans we apply mode 
    def is_integer(series):
        non_na = series.dropna()
        return ((non_na % 1) == 0).all()
    for col in df.select_dtypes(include='number'):
        if is_integer(df[col]):
            df[col].fillna(df[col].mode()[0], inplace=True)
            df[col] = df[col].astype(int)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object'):
        if df[col].isna().mean() * 100 < 20:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna("missing", inplace=True)
    return df    

# --- Step 3: Encode categoricals ---
class EncodeCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.encoders = {}
        for col in X.select_dtypes(include='object'):
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            enc.fit(X[[col]])
            self.encoders[col] = enc
        return self
    def transform(self, X):
        df = X.copy()
        for col, enc in self.encoders.items():
            transformed = enc.transform(df[[col]])
            new_cols = [f"{col}_{cat}" for cat in enc.categories_[0]]
            for i, new_col in enumerate(new_cols):
                df[new_col] = transformed[:, i]
            df.drop(columns=[col], inplace=True)
        return df

# --- Step 4: Scale numerics ---
class ScaleNumerics(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.scaler = StandardScaler()
        self.num_cols = X.select_dtypes(include='number').columns
        self.scaler.fit(X[self.num_cols])
        return self
    def transform(self, X):
        df = X.copy()
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


class AnalyzerTransformer:
    def __init__(self, outlier_threshold=0.05):
        self.outlier_threshold = outlier_threshold
        self.scalers = {}
        self.encoders = {}
        self.decision_log = {}

    def fit(self, df):
        transformed_df = df.copy()

        for col in df.columns:
            data = df[col]

            # Skip if constant
            if data.nunique() <= 1:
                continue

            # Numeric features
            if np.issubdtype(data.dtype, np.number):
                col_skew = skew(data, nan_policy="omit")
                col_kurt = kurtosis(data, fisher=True, bias=False, nan_policy="omit")
                outlier_ratio = (np.abs((data - data.mean()) / data.std()) > 3).mean()

                decision_steps = []
                transformed = data.values.reshape(-1, 1)

                # --- Step 1: Handle skew ---
                if abs(col_skew) > 1 and data.std() > 0:
                    try:
                        transformer = PowerTransformer(method="yeo-johnson")
                        transformed = transformer.fit_transform(transformed)
                        self.scalers[col] = transformer
                        decision_steps.append("PowerTransform")
                    except Exception as e:
                        print(f"Skipping PowerTransform for {col}: {e}")

                    # recompute after transform
                    col_skew = skew(transformed.flatten())
                    col_kurt = kurtosis(transformed.flatten(), fisher=True, bias=False)

                # --- Step 2: Handle outliers / scaling ---
                if abs(col_kurt) > 2 or outlier_ratio > self.outlier_threshold:
                    scaler = RobustScaler()
                    transformed = scaler.fit_transform(transformed)
                    self.scalers[col] = scaler
                    decision_steps.append("RobustScaler")
                elif abs(col_skew) < 0.5 and abs(col_kurt) < 1:
                    scaler = StandardScaler()
                    transformed = scaler.fit_transform(transformed)
                    self.scalers[col] = scaler
                    decision_steps.append("StandardScaler")
                else:
                    scaler = MinMaxScaler()
                    transformed = scaler.fit_transform(transformed)
                    self.scalers[col] = scaler
                    decision_steps.append("MinMaxScaler")

                transformed_df[col] = transformed.flatten()

            # Categorical features
            else:
                unique_vals = data.nunique()
                decision_steps = []

                if unique_vals <= 10:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    transformed = encoder.fit_transform(data.values.reshape(-1, 1))
                    new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    for i, new_col in enumerate(new_cols):
                        transformed_df[new_col] = transformed[:, i]
                    transformed_df.drop(columns=[col], inplace=True)
                    self.encoders[col] = encoder
                    decision_steps.append("OneHotEncoder")
                else:
                    encoder = LabelEncoder()
                    transformed_df[col] = encoder.fit_transform(data)
                    self.encoders[col] = encoder
                    decision_steps.append("LabelEncoder")

            self.decision_log[col] = decision_steps

        return transformed_df

    def transform(self, df):
        transformed_df = df.copy()

        for col in df.columns:
            if col in self.scalers:  # numeric
                transformed = transformed_df[col].values.reshape(-1, 1)
                transformed = self.scalers[col].transform(transformed)
                transformed_df[col] = transformed.flatten()

            elif col in self.encoders:  # categorical
                encoder = self.encoders[col]

                if isinstance(encoder, OneHotEncoder):
                    transformed = encoder.transform(transformed_df[col].values.reshape(-1, 1))
                    new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    for i, new_col in enumerate(new_cols):
                        transformed_df[new_col] = transformed[:, i]
                    transformed_df.drop(columns=[col], inplace=True)

                elif isinstance(encoder, LabelEncoder):
                    # Handle unseen labels
                    transformed_df[col] = transformed_df[col].map(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )

        return preprocessed_df

    def fit_transform(self, df):
        return self.fit(df)
