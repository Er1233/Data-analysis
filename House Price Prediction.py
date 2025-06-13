from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

california = fetch_california_housing(as_frame=True)
df = california.frame

x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2)

model = RandomForestRegressor(random_state=42)

model.fit(x_train, y_train)

pred = model.predict(x_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"mean squared error {mse:.2f}")
print(f"r2 score {r2:.3f}")

print("Data visualization..... \n")
plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices Scatter")
plt.grid(True)
plt.show()


sample_idx = 0
actual_price = y_test.iloc[sample_idx]
predicted_price = pred[sample_idx]
print(f"\nExample: Actual ${actual_price*100000:.0f} vs Predicted ${predicted_price*100000:.0f}")