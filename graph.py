import pandas as pd
import openpyxl
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# загрузка данных из Excel
data = pd.read_excel('nerual.xlsx')

# разделение данных на тренировочный и тестовый наборы
train_data, test_data = train_test_split(data, test_size=0.2)

# определение модели нейронной сети
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# компиляция модели с метрикой accuracy и loss
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# обучение модели
history = model.fit(train_data.iloc[:, :8], train_data.iloc[:, 8], epochs=50, batch_size=32, validation_split=0.2)

# оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_data.iloc[:, :8], test_data.iloc[:, 8])

# прогнозирование результатов на новых данных
new_data = pd.DataFrame({'Column 1': [1, 2, 3], 'Column 2': [4, 5, 6], 'Column 3': [7, 8, 9], 'Column 4': [10, 11, 12],
                         'Column 5': [13, 14, 15], 'Column 6': [16, 17, 18], 'Column 7': [19, 20, 21],
                         'Column 8': [22, 23, 24]})
predictions = model.predict(new_data.iloc[:, :8])
model.save('model.h5')

# построение графика точности
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# построение графика ошибок
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
