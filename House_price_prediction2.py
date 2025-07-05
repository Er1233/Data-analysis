from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("housing.csv")
print(df.columns)
print(df.isnull().sum())

#feature engineering
df['rooms_per_households'] = df['total_rooms'] / df['households']
df['population_per_household'] = df['population'] / df['households']
df['bedrooms_per_rooms'] = df['total_bedrooms'] / df['total_rooms']
df['bedrooms_per_households'] = df['total_bedrooms'] / df['households']

#handling missing values
imputer = SimpleImputer(strategy='median')
numerical_column = df.select_dtypes(include=[np.number],).columns
df[numerical_column] = imputer.fit_transform(df[numerical_column])

#encode categorical variable
df_encode = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', drop_first=True)

#prepare x and y
x = df_encode.drop('median_house_value', axis=1)
y = df_encode['median_house_value']

#log transform
print(f"Log transform target: {y.skew():.2f}")
if y.skew() > 1:
    y = np.log(y)
    print("Applied log transformation to target")

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

#scale
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_test_scale = scaler.transform(x_test)

#change back to a dataframe
x_train_scale = pd.DataFrame(x_train_scale, columns=x.columns, index=x_train.index)
x_test_scale = pd.DataFrame(x_test_scale, columns=x.columns, index=x_test.index)

#model
model = RandomForestRegressor(
    n_estimators=100,
    n_jobs=1,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)

#model fit
model.fit(x_train_scale, y_train)

#model Prediction
y_pred = model.predict(x_test_scale)

#metrics calculation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mse {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 score: {r2:.4f}")

#sample prediction
print("creating Sample prediction.....")

for i in range(3):
    actual = y_test.iloc[i]
    predict = y_pred[i]
    error = abs(actual - predict)
    error_pct = (error / actual)*100
    print(f'Sample {i +1}')
    print(f"Actual {actual:.4f}")
    print(f"prediction {predict:.4f}")
    print(f"errors: {error:.2f}, {error_pct:.1f}%")
    print()