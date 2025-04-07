import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Пример структуры
df = pd.read_csv("your_data.csv")

# Колонки: features, y_predict, remains, y_gt
features = [col for col in df.columns if col.startswith("feature_")]
X = df[features]
y_predict = df["y_predict"]
remains = df["remains"]
y_gt = df["y_gt"]

# Вычисляем дельту
df["delta_y"] = df["y_gt"].diff()
df.dropna(inplace=True)


X = df[features + ["y_predict"]]
y = df["delta_y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_diff = GradientBoostingRegressor()
model_diff.fit(X_train, y_train)

# Прогноз дельт
y_delta_pred = model_diff.predict(X_test)

# Восстанавливаем значение: y_t = y_{t-1} + delta
y_base = df["y_gt"].iloc[X_test.index - 1].values
y_reconstructed = y_base + y_delta_pred
