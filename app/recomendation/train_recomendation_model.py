import sqlite3
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
import joblib

# Шаг 1: Подключение к базе данных и получение данных
def get_data_from_db():
    # из бд reviews нужно получить user_id, place_id, rating
    pass

# Шаг 2: Подготовка данных для обучения
def prepare_data_for_training(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'place_id', 'rating']], reader)
    return data

# Шаг 3: Обучение модели
def train_model(data):
    # Разделение данных на тренировочную и тестовую выборки
    trainset, testset = train_test_split(data, test_size=0.2)

    # Обучаем модель SVD
    model = SVD()
    model.fit(trainset)

    # Оценка точности на тестовых данных
    predictions = model.test(testset)
    print(f"RMSE: {accuracy.rmse(predictions)}")

    return model

# Шаг 4: Сохранение обученной модели
def save_model(model, model_filename='svd_model.pkl'):
    # Сохраняем модель с помощью joblib
    joblib.dump(model, model_filename)
    print(f"Модель сохранена как {model_filename}")

# Шаг 5: Главная функция для всего процесса
def main():
    # Получаем данные из базы данных
    df = get_data_from_db()

    # Подготавливаем данные для обучения
    data = prepare_data_for_training(df)

    # Обучаем модель
    model = train_model(data)

    # Сохраняем модель
    save_model(model)

# Запускаем процесс
if __name__ == "__main__":
    main()
