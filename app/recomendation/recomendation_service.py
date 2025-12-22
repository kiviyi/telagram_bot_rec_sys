from surprise import SVD, Reader, Dataset
from surprise import accuracy
import pandas as pd
import uuid
import joblib
# Вариант 1: Загрузить сохраненную модель
model = joblib.load('svd_model.pkl')
## Получаем список подходящий по тегам заведений
'''
SELECT 
    UUID() AS id,  -- Генерация уникального идентификатора (например, UUID)
    l.name, 
    l.description, 
    l.city, 
    l.price_level, 
    AVG(r.rating) AS rating_avg,  -- Средний рейтинг
    COUNT(r.id) AS rating_cnt    -- Количество отзывов
FROM locations l
LEFT JOIN reviews r ON l.id = r.location_id
WHERE l.city = 'Москва'
  AND l.price_level = 3
  AND EXISTS (
      SELECT 1 
      FROM tags t 
      WHERE t.tag_name IN ('кофейня', 'можно с ноутбуком', 'тихо') 
      AND FIND_IN_SET(t.tag_name, l.tags) > 0
  )
  AND NOT EXISTS (
      SELECT 1
      FROM tags t
      WHERE t.tag_name = 'бар' 
      AND FIND_IN_SET(t.tag_name, l.tags) > 0
  )
GROUP BY l.id
ORDER BY rating_avg DESC;

'''
df = pd.read_json('ratings_data.json')

# Функция для предсказания рейтинга для пользователя и места
def predict_rating(user_id, place_id):
    return model.predict(user_id, place_id).est

# Функция для обработки холодного старта
def is_cold_start(place_id):
    # Если у места меньше 5 оценок, это холодный старт
    place_ratings = df[df['place_id'] == place_id]
    return len(place_ratings) < 5

# Функция для сортировки по популярности (если холодный старт)
def sort_by_popularity(candidates):
    # Сортировка по количеству отзывов или среднему рейтингу
    return sorted(candidates, key=lambda x: (x['rating_cnt'], x['rating_avg']), reverse=True)

# Функция для получения рекомендаций с учетом холодного старта
def rank_places(user_id, candidates):
    ranked_places = []

    for place in candidates:
        place_id = place['id']
        if is_cold_start(place_id):
            # Если холодный старт, сортируем по популярности
            ranked_places.append({
                'id': place_id,
                'name': place['name'],
                'predicted_rating': None,  # нет предсказания, только сортировка по популярности
                'rating_avg': place['rating_avg'],
                'rating_cnt': place['rating_cnt']
            })
        else:
            # Для места с достаточным количеством оценок, используем модель для предсказания
            predicted_rating = predict_rating(user_id, place_id)
            ranked_places.append({
                'id': place_id,
                'name': place['name'],
                'predicted_rating': predicted_rating,
                'rating_avg': place['rating_avg'],
                'rating_cnt': place['rating_cnt']
            })

    # Сортировка по предсказанному рейтингу (если холодного старта нет)
    ranked_places = sorted(ranked_places, key=lambda x: (x['predicted_rating'] if x['predicted_rating'] is not None else 0), reverse=True)
    
    return ranked_places

# Пример кандидатов (мест)
candidates = [
    {"id":"22222222-2222-2222-2222-222222222222","name":"Dessert House","description":"десерты и латте","city":"Moscow","price_level":3,"rating_avg":4.8,"rating_cnt":15},
    {"id":"33333333-3333-3333-3333-333333333333","name":"Noisy Coffee","description":"кофе и латте, громко","city":"Moscow","price_level":2,"rating_avg":4.7,"rating_cnt":300},
    {"id":"11111111-1111-1111-1111-111111111111","name":"Coffee Lab","description":"спешелти кофе, латте","city":"Moscow","price_level":2,"rating_avg":4.6,"rating_cnt":120},
    {"id":"44444444-4444-4444-4444-444444444444","name":"Latte & Co","description":"латте, тихо","city":"Moscow","price_level":3,"rating_avg":4.5,"rating_cnt":80},
    {"id":"55555555-5555-5555-5555-555555555555","name":"Central Roasters","description":"спешелти кофе, тихо","city":"Moscow","price_level":3,"rating_avg":4.4,"rating_cnt":40},
    {"id":"66666666-6666-6666-6666-666666666666","name":"Art Cafe","description":"кофе и десерты","city":"Moscow","price_level":2,"rating_avg":4.3,"rating_cnt":60},
    {"id":"77777777-7777-7777-7777-777777777777","name":"Morning Cup","description":"латте и выпечка","city":"Moscow","price_level":2,"rating_avg":4.1,"rating_cnt":20},
    {"id":"88888888-8888-8888-8888-888888888888","name":"Hidden Yard","description":"тихая кофейня, латте","city":"Moscow","price_level":2,"rating_avg":4.8,"rating_cnt":5},
    {"id":"99999999-9999-9999-9999-999999999999","name":"Library Coffee","description":"тихо, много места, латте","city":"Moscow","price_level":2,"rating_avg":4.6,"rating_cnt":200},
    {"id":"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa","name":"Brew & Talk","description":"кофе, разговоры","city":"Moscow","price_level":2,"rating_avg":4.2,"rating_cnt":35}
]

# Пример user_id для одного пользователя
user_id = uuid.uuid4()  # Например, уникальный идентификатор пользователя

# Получаем рекомендованные места
ranked_places = rank_places(user_id, candidates)

# Выводим отранжированные места
print("Рекомендации для пользователя:")
for place in ranked_places:
    print(f"{place['name']} (Предсказанный рейтинг: {place['predicted_rating'] if place['predicted_rating'] else 'N/A'})")


with open('Recomndation_output.json', 'w', encoding='utf-8') as f:
    json.dump(ranked_places, f, ensure_ascii=False, indent=2)