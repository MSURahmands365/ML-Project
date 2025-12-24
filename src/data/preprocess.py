import pandas as pd
def preprocess_data(df:pd.DataFrame,target_col:str="Price")->pd.DataFrame:
  df.columns=df.columns.str.strip()
  df = df.drop(columns=['Unnamed: 0'], errors='ignore')  
  if target_col in df.columns:
    df['Price']=df['Price'].astype(str).str.replace(',','').str.extract('(\d+)',expand=False)
    df['Price']=pd.to_numeric(df['Price'],errors='coerce')
    df=df[df['Price']>1000]
  if 'Engine Capacity(CC)' in df.columns:
    df['Engine Capacity(CC)']=df['Engine Capacity(CC)'].astype(str).str.extract('(\d+)',expand=False)
    df['Engine Capacity(CC)']=pd.to_numeric(df['Engine Capacity(CC)'],errors='coerce')
    
  if 'Year' in df.columns:
    df['Year']=pd.to_numeric(df['Year'],errors='coerce')  
    df=df[(df['Year']>1980 ) & (df['Year']<=2025)]
  
  if 'City' in df.columns:
    df['City']=df['City'].str.strip().str.title()
     
  return df  
    
           
    
  
  
  

  