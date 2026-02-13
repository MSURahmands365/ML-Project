import pandas as pd 
import category_encoders as ce 
import joblib


def fit_and_save_encoders(df: pd.DataFrame, target_col: str = "Price"):
    
    # 1. Setup
    target_enc = ce.TargetEncoder(cols=['Name', 'City'], smoothing=10)
    ohe_enc = ce.OneHotEncoder(cols=['Make', 'Transmission', 'Engine Type'], use_cat_names=True)
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 2. Fit
    target_enc.fit(X[['Name', 'City']], y)
    ohe_enc.fit(X[['Make', 'Transmission', 'Engine Type']])
    
    # 3. Save Encoders
    joblib.dump(target_enc, 'target_encoder.pkl')
    joblib.dump(ohe_enc, 'one_hot_encoder.pkl')

    # 4. MERGE DIRECTLY INTO DF
    # First, handle Target Encoding (replaces the text in 'Name' and 'City' with numbers)
    df[['Name', 'City']] = target_enc.transform(df[['Name', 'City']])
    
    # Second, handle One-Hot Encoding (this adds NEW columns like Make_Honda, etc.)
    df_ohe_only = ohe_enc.transform(df[['Make', 'Transmission', 'Engine Type']])
    
    # Drop the old text columns and join the new binary columns
    df.drop(columns=['Make', 'Transmission', 'Engine Type'], inplace=True)
    for col in df_ohe_only.columns:
        df[col] = df_ohe_only[col]

    print("All columns merged into original df successfully.")
    return df


def apply_encoders(df: pd.DataFrame, target_enc, ohe_enc):
    """
    Transforms a dataframe using already fitted encoders.
    Used for both Test data and UI input.
    """
    df_ohe = ohe_enc.transform(df)
    df_final = target_enc.transform(df_ohe)
    return df_final
  
  