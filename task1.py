import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier  
from sklearn.tree import DecisionTreeRegressor   
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('restaurant.csv') # Using a raw string for the local path

print("First 5 rows of dataset:")
display(df.head())
print("\nDataset Info:")
df.info()


df = df.dropna()  

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
target_column = "Aggregate rating"
x = df.drop(target_column,axis=1)
y=df[target_column]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
prediction = x_test.iloc[0:1]
predicted_rating = model.predict(prediction)
print("Input:",prediction)
print("predicted Rating:",predicted_rating)
