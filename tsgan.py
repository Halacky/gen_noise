import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.synthesizers.timeseries import TimeGAN
from ydata_synthetic.preprocessing.timeseries import processed_stock
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv('your_dataset.csv')

# Выделение признаков и целевой переменной
features = data[['feature_1', 'feature_2', ..., 'y_predict', 'y_gt']].values
target = data['remains'].values

# Нормализация данных
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.reshape(-1, 1))

# Объединение признаков и целевой переменной
scaled_data = np.concatenate([scaled_features, scaled_target], axis=1)

# Преобразование данных в формат, подходящий для TimeGAN (последовательности)
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 10  # Длина временного окна
sequences = create_sequences(scaled_data, seq_length)

# Разделение на train и test
X_train, X_test = train_test_split(sequences, test_size=0.2, random_state=42)

# Параметры модели
hidden_dim = 24  # Количество нейронов в скрытых слоях
num_layers = 3   # Количество слоев в RNN
iterations = 5000  # Количество итераций обучения

# Создание и обучение TimeGAN
gan = TimeGAN(
    model_parameters={
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'batch_size': 128,
        'module_name': 'gru',  # или 'lstm'
        'seq_len': seq_length,
        'n_features': scaled_data.shape[1],
        'gamma': 1
    }
)

gan.train(
    X_train,
    train_steps=iterations,
    save_synthetic=True,
    save_dir='./models'
)

# Генерация синтетических данных
synthetic_data = gan.sample(len(X_test))

# Извлечение предсказаний для remains (последний столбец)
synthetic_remains = synthetic_data[:, -1, -1]  # Берем последний временной шаг и последний признак

# Обратное преобразование масштаба
predicted_remains = scaler.inverse_transform(synthetic_remains.reshape(-1, 1))

# Для реальных данных (из тестовой выборки)
real_remains = X_test[:, -1, -1]  # Последний временной шаг, последний признак
real_remains = scaler.inverse_transform(real_remains.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(real_remains, prediced_remains)
mse = mean_squared_error(real_remains, prediced_remains)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# # Генерация дополнительных синтетических данных
# synthetic_augment = gan.sample(5000)  # Генерируем 5000 примеров

# # Подготовка данных для классической модели
# X_train_augmented = np.concatenate([X_train, synthetic_augment])
# y_train_augmented = X_train_augmented[:, -1, -1]  # remains - последний столбец
# X_train_augmented = X_train_augmented.reshape(-1, seq_length * scaled_data.shape[1])

# # Обучение, например, RandomForest
# from sklearn.ensemble import RandomForestRegressor

# rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X_train_augmented, y_train_augmented)

# # Подготовка тестовых данных
# X_test_flat = X_test.reshape(-1, seq_length * scaled_data.shape[1])
# y_test = X_test[:, -1, -1]

# # Предсказание и оценка
# y_pred = rf.predict(X_test_flat)
# print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
