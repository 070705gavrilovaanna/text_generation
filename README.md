# Text Generation with LSTM and GPT-2

Проект по генерации текста с использованием двух архитектур: LSTM (обучена с нуля) и предобученного GPT-2. Модели обучались на корпусе цитат о любви и отношениях.

**Ссылка на проект:** [text_generation_lstm_gpt2.ipynb](https://colab.research.google.com/drive/your_link_here)

---

## Этапы работы

**Технологии:** Python, PyTorch, Transformers (Hugging Face), TensorFlow/Keras, Scikit-learn, NumPy

1. **Подготовка данных:** Загрузка текстового корпуса (цитаты о любви), токенизация через Keras Tokenizer (vocab_size=20000, oov_token='<unk>'), преобразование текста в последовательности чисел, создание обучающих пар (вход: 50 слов → следующее слово), ограничение до 200,000 примеров.

2. **Разделение данных:** X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42). Train/val: 80/20, batch_size=128.

3. **Архитектура LSTM:** Embedding(20000,128) → LSTM(128,256,dropout=0.1) → Dropout(0.1) → Linear(256,20000).

4. **Обучение:** CrossEntropyLoss, Adam(lr=0.001, weight_decay=1e-5), 15 эпох с сохранением лучшей модели по val_loss. Лучшая эпоха: 10 (val_loss=5.48).

5. **Генерация текста (LSTM):** Функция generate_text с параметрами temperature (0.5-1.2), top_k (40-100) и принудительным обнулением токена <unk>.

6. **Сравнение с GPT-2:** Загрузка предобученной модели GPT-2 через Hugging Face, генерация с аналогичными параметрами.

---

# Результаты генерации

## LSTM (temperature=0.8, top_k=100):

Love is: Love is individual and beautiful and yet if i should be no enough for you for the person who carries me and

I am: I am someone about it i was saying that the people who have drunk my life for sure of life i was

What: What lose ourselves for the way or whether i will be an extremely human being to put up in the morning

Living on: Living on ingredients up sometimes of a person you know if you think he stepped up and the way she wanted to

People are: People are inside their presence it was just a time but the only thing i was as though in the power to



## GPT-2 (temperature=0.9):

Love is: Love is my favorite brand. I found your blog so interesting, I wanted to share it with you.

I am: I am not saying this is wrong. For what reason they did it? They had the power it granted to them.

What: What the heck is a 'R' when it comes to a 'R'?" asked Joe Kors on The Dave Brubeck Show.

Living on: Living on vacation with my brother, I made some of the food, got some of the clothes, and then we took a walk on the beach.

People are: People are the ones that have been doing this for long enough that it seems natural to believe that it really is all about you.



---

## Сравнение моделей

| | LSTM | GPT-2 |
|---|---|---|
| Размер | 10 MB | 500 MB |
| Обучение | С нуля на цитатах | Предобучена на 40GB текста |
| Что умеет | Пишет в стиле цитат | Пишет связно на разные темы |
| Стиль | Подстраивается под мой текст | Универсальный |
| Качество | Бывает сбой | Стабильно хорошее |

---

## Что я узнала

1. Валидация помогла выбрать лучшую эпоху и избежать переобучения.
2. Dropout=0.1 + weight_decay=1e-5 дали стабильное обучение.
3. Top-K и temperature критически влияют на качество генерации.
4. LSTM выучила стиль цитат, GPT-2 дает более естественные тексты.
5. Обнуление `<unk>` значительно улучшило качество генерации.
