# User's actions classifier

Приложение для классификации действий пользователя (опасные / неопасные)

Запуск:
1. Отредактировать файл .env - добавить ключ API для получения и загрузки данных (строка вида: 'xxxxxxx')
2. Установить необходимые пакеты **pip install -r requirements.txt** (Для работы GPU PyTorch дополнительно требуется CUDA 11.3)
3. Запустить приложение: **streamlit run stapp.py**

Основные вкладки:  
* **stapp**  - запуск процесса получения новых векторов и отправки предсказаний класса. Необходима предварительно обученная модель (владка train model). В качестве параметров нужно выбрать размер загружаемого батча и количество итераций. *При получении ошибки во время загрузки приложение уходит в режим ожидания на 30 минут.*
* **current statistics** - просмотр текущей статистики пользователя. При нажатии на кнопку update загружаются последние данные (если менялось число отправленных векторов). Для отображения нового графика нужно обновить страницу.
* **train model** - запуск обучения модели на текущих данных из таблицы 'vectors'

Описание работы:  
Для получения векторного представления поступащих данных используется tf-idf vectorizer  
Для предсказания - комбинация моделей autoencoder (для понижения размерности) и kmeans (для кластеризации)  
Хранение данных с помощью SQLite. При загрузке векторов они сохраняются в таблицу 'vectors', которая также используется для обучения модели. Список обученных моделей - таблица 'models' и история статистики - таблица 'stats'

Точность модели: 77.7399 %, отправлено за попытку: 1469 векторов, сложность 3

![screen](https://github.com/DDarean/ml-comp/blob/readme/data/screen.PNG)
