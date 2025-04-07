import pywt
import numpy as np

# Вейвлет-разложение на уровне 2 (можно менять)
coeffs = pywt.wavedec(df["y_gt"], 'db4', level=2)
cA2, cD2, cD1 = coeffs

# Предположим, что cA2 — тренд (не трогаем), а детали предсказываем
detail_len = len(cD2)  # Детали короче из-за паддинга

# Создаём фичи с учетом длины деталей
X_detail = df[features + ["y_predict"]].iloc[-detail_len:].reset_index(drop=True)
y_detail = pd.Series(cD2)

model_detail = GradientBoostingRegressor()
model_detail.fit(X_detail, y_detail)

# Предсказание деталей
y_detail_pred = model_detail.predict(X_detail)

# Собираем обратно: используем исходный тренд и новые детали
coeffs_new = [cA2, y_detail_pred, cD1]  # заменяем только cD2
y_reconstructed = pywt.waverec(coeffs_new, 'db4')
