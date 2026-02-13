run pipeline for training steps

python scripts/run_pipeline.py pipeline `
  --input data/raw/PakWheelsDataSet.csv `
  --target Price `
  --test_size 0.2 `
  --experiment "PakWheels Price Prediction" `
  --tune

to check the id of running pipe line 
  mlflow ui --backend-store-uri sqlite:///mlflow.db

prediction 

python scripts/run_pipeline.py predict `
  --input data/new_inputs.csv `
  --model_uri runs:/<RUN_ID>/model `
  --output predictions.csv
