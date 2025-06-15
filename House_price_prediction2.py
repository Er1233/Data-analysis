from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

df = pd.read_csv("housing.csv")
print("Original data shape:", df.shape)
print("\n missing values")
print(df.isnull().sum())
print("\n data type: ")
print(df.dtypes)
 #feature Engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_rooms"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]
df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]

#handling missing values

imputer = SimpleImputer(strategy="median")
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

#encode categorical variable

df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean", drop_first=True)

#prepare x and y

x = df_encoded.drop(["median_house_value"], axis=1)
y  =df_encoded["median_house_value"]

#check skewness first
print(f"\n Target variable skewness: {y.skew():.2f}")
if y.skew() > 1:
    y = np.log(y)
    print("Applied log transformation to target variable")

#train-test-split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

#feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#convert back to DataFrame for easier handling

x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns, index=x_train.index)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns, index=x_test.index)

#model training with better hyperparameters

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1

)
# fit model
model.fit(x_train_scaled, y_train)

#prediction and Evaluation
y_pred = model.predict((x_test_scaled))

#calculate metrics

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n===MODEL PERFORMANCE ===")
print(f"R2 score: {r2:.4f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")


