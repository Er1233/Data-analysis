#Ml stock prediction model
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = yf.download("^GSPC", start="2023-01-01", end="2025-06-01")
data = data.dropna()

data["Yesterday_close"] = data["Close"].shift(1)
data["Today_close"] = data["Close"]

data = data.dropna()

data["Direction"] = (data["Today_close"] > data["Yesterday_close"]).astype(int)

data["Daily_return"] = data["Close"].pct_change()
data["MA_5"] = data["Close"].rolling(5).mean()
data["MA_10"] = data["Close"].rolling(10).mean()

data["Volatility"] = data["Daily_return"].rolling(10).std()

data = data.dropna()

x = data[["Yesterday_close","MA_5","MA_10", "Daily_return","Volatility"]]
y = data["Direction"]

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=False, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)


print("Accuracy", model.score(x_test, y_test))

predictions = model.predict(x_test)
print("Classification_report", classification_report(y_test, predictions))
print("Confusion_metric", confusion_matrix(y_test, predictions))

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual values")
plt.plot(predictions, label="Predicted values", linestyle="--")
plt.legend()
plt.title("Actual vs Predicted values")
plt.show()

