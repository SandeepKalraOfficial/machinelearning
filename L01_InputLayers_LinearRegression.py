import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error,mean_squared_error

import random
random.seed(42)  
np.random.seed(42)          # seed For NumPy operations
tf.random.set_seed(42)      # seed for tensorflow operations

X=np.array([
  [3,1500,2,2010],
  [2,1200,1,2005],
  [4,2000,3,2018],
  [3,1600,2,2012]
],dtype=np.float32)



y=np.array([300,200,400,320],dtype=np.float32)
columns =['Bedrooms','Area','Bathrooms','Built Year']

df=pd.DataFrame(X,columns=columns)
df['Price']=y

plt.figure(figsize=(12,4))
#Bar plot for each feature
for i, col in enumerate(columns):
  plt.subplot(1,4,i+1)
  plt.bar(df.index,df[col])
  plt.title(col)
  plt.xticks(df.index)
  plt.grid(True)

plt.tight_layout()
plt.suptitle('Input Feature Distribution',y=1.1)
plt.show()

#Scatter Plot Area VS Price
plt.figure(figsize=(5,4))
plt.scatter(df['Area'],df['Price'],c='blue',marker='o')
plt.title("Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($1000s)")
plt.grid(True)
plt.tight_layout()
plt.show()

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(4,)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dense(4, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=300,verbose=1)

#Predictions and Evaluations
predictions=model.predict(X).flatten()
mae=mean_absolute_error(y,predictions)
mse=mean_squared_error(y,predictions)
rmse=root_mean_squared_error(y,predictions)
r2=r2_score(y,predictions)

print("\n📈 Evaluation Metrics:")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")


plt.figure(figsize=(6,4))
plt.plot(y,label='True Price', marker='o')
plt.plot(predictions,label='Predicted Prices', marker='x')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price ($1000s)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

model_path='Models/L01_Input_Layers_LinearRegression.keras'
model.save(model_path)
print('Model saved successfully')

loaded_model = tf.keras.models.load_model(model_path)
new_predictions=loaded_model.predict(X).flatten()
print(f"Predictions from loaded model:{new_predictions}")
