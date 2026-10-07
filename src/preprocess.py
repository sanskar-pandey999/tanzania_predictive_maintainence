import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_df(df, is_train=True, encoders=None):
    """
    Cleans and preprocesses feature DataFrames for both training and inference.
    """
    df = df.copy()
    
    # Drop redundant, zero-variance, or high-cardinality ID columns
    cols_to_drop = [
        'recorded_by', 'wpt_name', 'subvillage', 'ward', 
        'scheme_name', 'num_private', 'region_code', 'district_code'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Handle missing / invalid geographic coordinates
    if 'longitude' in df.columns:
        df['longitude'] = df['longitude'].replace(0.0, np.nan)
        if 'region' in df.columns:
            df['longitude'] = df['longitude'].fillna(df.groupby('region')['longitude'].transform('mean'))
            
    if 'latitude' in df.columns:
        df['latitude'] = df['latitude'].replace(0.0, np.nan)
        if 'region' in df.columns:
            df['latitude'] = df['latitude'].fillna(df.groupby('region')['latitude'].transform('mean'))
            
    if 'gps_height' in df.columns:
        df['gps_height'] = df['gps_height'].replace(0, np.nan)
        if 'region' in df.columns:
            df['gps_height'] = df['gps_height'].fillna(df.groupby('region')['gps_height'].transform('median'))
            
    if 'construction_year' in df.columns:
        df['construction_year'] = df['construction_year'].replace(0, np.nan)
        if 'extraction_type' in df.columns:
            df['construction_year'] = df['construction_year'].fillna(
                df.groupby('extraction_type')['construction_year'].transform('median')
            )

    # Impute missing numerical values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median() if not np.isnan(df[col].median()) else 0)

    # Clean text categorical fields
    if 'installer' in df.columns:
        df['installer'] = df['installer'].fillna('Unknown').astype(str).str.lower()
        df['installer'] = df['installer'].apply(
            lambda x: 'dwe' if 'dwe' in x else ('government' if 'gov' in x else 'other')
        )
        
    if 'funder' in df.columns:
        df['funder'] = df['funder'].fillna('Unknown').astype(str).str.lower()
        df['funder'] = df['funder'].apply(
            lambda x: 'danida' if 'danida' in x else ('gov' if 'gov' in x else 'other')
        )

    # Extract date components
    if 'date_recorded' in df.columns:
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        df['year_recorded'] = df['date_recorded'].dt.year
        df['month_recorded'] = df['date_recorded'].dt.month
        df = df.drop(columns=['date_recorded'])

    # Categorical Label Encoding
    if encoders is None:
        encoders = {}
        
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        if col == 'status_group':
            continue
        df[col] = df[col].astype(str)
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                # Handle unseen category levels during inference
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                
    return df, encoders