from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import numpy as np
def  evaluate_model(model,x_test,y_test):
  preds=model.predict(x_test)
  mae=mean_absolute_error(y_test,preds)
  rmse=np.sqrt(mean_squared_error(y_test,preds))
  r2=r2_score(y_test,preds)
  print("\n Regression Evaluation Report:")
  print("-" * 30)
  print(f"Mean Absolute Error (MAE):  PKR {mae:,.2f}")
  print(f"Root Mean Squared Error:   PKR {rmse:,.2f}")
  print(f"R2 Score (Accuracy):       {r2:.4f}") 
  