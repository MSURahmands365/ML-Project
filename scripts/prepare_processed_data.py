import os,sys
import pandas as pd 
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from src.data.preprocess import preprocess_data
from src.features.build_features import fit_and_save_encoders
RAW="data/raw/PakWheelsDataSet.csv"
OUT="data/processed/processedPakWheelsDataSet.csv"

df=pd.read_csv(RAW)
important_cols=['Make', 'Name', 'Transmission', 'Engine Type',
       'Engine Capacity(CC)', 'City', 'Year', 'Price' ]
df.dropna(subset=important_cols,inplace=True)
df=preprocess_data(df,target_col='Price')

if "Price" in df.columns and df["Price"].dtype == "object":
  df['Price']=pd.to_numeric(df['Price'],errors='coerce')
  
df=df[df['Price']>100000]
df.drop_duplicates(inplace=True)

df_processed=fit_and_save_encoders(df,target_col='Price')

te = joblib.load('target_encoder.pkl')
name_mapping = te.mapping['Name']

print(df_processed.columns.tolist())
print(df_processed.info())

os.makedirs(os.path.dirname(OUT),exist_ok=True)
df_processed.to_csv(OUT,index=False)
print(f"Processed dataset saved to {OUT} | Shape: {df_processed.shape}")