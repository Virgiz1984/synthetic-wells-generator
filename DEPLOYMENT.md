# 🚀 Инструкции по деплою

## 📋 Создание репозитория на GitHub

### 1. Создайте новый репозиторий на GitHub:
1. Перейдите на [GitHub.com](https://github.com)
2. Нажмите кнопку "New repository" или "+" → "New repository"
3. Заполните поля:
   - **Repository name**: `synthetic-wells-generator`
   - **Description**: `Streamlit app for generating synthetic well logs using Markov chains and transgression/regression cycles`
   - **Visibility**: Public (для бесплатного деплоя на Streamlit Cloud)
   - **Initialize**: НЕ ставьте галочки (у нас уже есть файлы)

### 2. Подключите локальный репозиторий к GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/synthetic-wells-generator.git
git branch -M main
git push -u origin main
```

## 🌐 Деплой на Streamlit Cloud

### 1. Подготовка к деплою:
- Убедитесь, что репозиторий публичный
- Файл `requirements.txt` должен содержать все зависимости
- Главный файл должен называться `app.py`

### 2. Деплой через Streamlit Cloud:
1. Перейдите на [share.streamlit.io](https://share.streamlit.io)
2. Войдите через GitHub аккаунт
3. Нажмите "New app"
4. Заполните поля:
   - **Repository**: `YOUR_USERNAME/synthetic-wells-generator`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `synthetic-wells-generator` (или любое свободное имя)

### 3. Настройки деплоя:
- **Python version**: 3.9 (автоматически)
- **Dependencies**: автоматически из `requirements.txt`

## 🔗 Альтернативные платформы для деплоя

### Heroku
```bash
# Создайте Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Добавьте в requirements.txt
echo "gunicorn" >> requirements.txt
```

### Railway
1. Подключите GitHub репозиторий
2. Railway автоматически определит Streamlit приложение
3. Настройте переменные окружения при необходимости

### Render
1. Создайте новый Web Service
2. Подключите GitHub репозиторий
3. Настройки:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 📱 Мобильная версия

Для улучшения мобильного опыта добавьте в `app.py`:
```python
# В начале файла после импортов
st.set_page_config(
    page_title="Генератор синтетических скважин",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## 🔧 Настройка домена (опционально)

### Для Streamlit Cloud:
- В настройках приложения можно настроить кастомный домен
- Требуется верификация домена

### Для других платформ:
- Настройте DNS записи
- Добавьте SSL сертификат

## 📊 Мониторинг

### Streamlit Cloud:
- Встроенная аналитика
- Логи доступны в панели управления

### Другие платформы:
- Настройте мониторинг через встроенные инструменты
- Добавьте логирование в приложение

## 🚨 Troubleshooting

### Частые проблемы:
1. **Ошибки импорта**: проверьте `requirements.txt`
2. **Медленная загрузка**: оптимизируйте код, используйте кэширование
3. **Ошибки памяти**: уменьшите размер данных или используйте пагинацию

### Логи:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 📈 Оптимизация производительности

### Кэширование:
```python
@st.cache_data
def generate_well_data(parameters):
    # Ваша функция генерации
    pass
```

### Ленивая загрузка:
```python
if st.button("Generate"):
    # Генерация только по запросу
    pass
```
