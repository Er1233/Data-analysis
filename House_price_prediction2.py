from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("housing.csv")
df = df.dropna()
print(df.columns)
print(df.dtypes)
print(df.head())

x = df.drop("ocean_proximity", axis=1)
y = df["median_house_value"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print("Model prediction", pred)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean squared error: {mse:.2f}")
print(f"r2 score: {r2:.2f}")

print("Data visualization __----")

plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.grid(True)
plt.show()

#Example case

sample = 0
actual = y_test.iloc[sample]
predicted = pred[sample]

print(f"Example predicted:actual price:  {actual:.2f} and the predicted price: {predicted:.2f}")