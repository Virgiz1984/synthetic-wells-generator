# 📊 Руководство по калибровке генератора на реальных данных

## 🎯 Цель
Извлечь статистические параметры из реальных скважин для генерации синтетических данных, максимально приближенных к реальности.

---

## 📋 Необходимые параметры для калибровки

### 1️⃣ **Литологические параметры**

#### 1.1 Частоты встречаемости литологий
Из реальных данных необходимо определить:
- **Процентное соотношение** каждого типа породы в разрезе
- **Минимальная и максимальная частота** для каждой литологии

**Как снять:**
```python
# Подсчет частот литологий
lithology_counts = df['Lithology'].value_counts()
lithology_frequencies = lithology_counts / len(df)

# Для нескольких скважин - определить диапазон
freq_range = {
    'sand': (min_freq_sand, max_freq_sand),
    'shale': (min_freq_shale, max_freq_shale),
    # и т.д.
}
```

#### 1.2 Длины литологических серий
- **Минимальная толщина** каждого типа породы (в метрах или точках)
- **Максимальная толщина** каждого типа породы
- **Средняя толщина** серий

**Как снять:**
```python
# Анализ длин серий
def analyze_series_lengths(lithology_column):
    series_lengths = {}
    current_lith = lithology_column.iloc[0]
    current_length = 1
    
    for lith in lithology_column.iloc[1:]:
        if lith == current_lith:
            current_length += 1
        else:
            if current_lith not in series_lengths:
                series_lengths[current_lith] = []
            series_lengths[current_lith].append(current_length)
            current_lith = lith
            current_length = 1
    
    # Статистика
    for lith in series_lengths:
        min_len = min(series_lengths[lith])
        max_len = max(series_lengths[lith])
        mean_len = np.mean(series_lengths[lith])
        print(f"{lith}: min={min_len}, max={max_len}, mean={mean_len:.1f}")
    
    return series_lengths
```

---

### 2️⃣ **Физические свойства пород**

#### 2.1 Пористость
Для каждой литологии определить:
- **Среднее значение** (μ)
- **Стандартное отклонение** (σ)
- **Минимум и максимум** (для проверки выбросов)

**Как снять:**
```python
for lith in unique_lithologies:
    lith_data = df[df['Lithology'] == lith]['Porosity']
    mean = lith_data.mean()
    std = lith_data.std()
    print(f"{lith}: μ={mean:.3f}, σ={std:.3f}")
```

#### 2.2 Водонасыщенность (Sw)
Аналогично пористости:
- Среднее значение
- Стандартное отклонение
- Диапазон значений

#### 2.3 Корреляции между параметрами
- **Porosity vs Sw** для каждой литологии
- **Porosity vs Density**
- **Gamma vs Lithology**

```python
# Корреляционная матрица по литологиям
for lith in unique_lithologies:
    lith_data = df[df['Lithology'] == lith]
    corr = lith_data[['Porosity', 'Sw', 'Gamma', 'Rho']].corr()
    print(f"\nКорреляция для {lith}:")
    print(corr)
```

---

### 3️⃣ **Каротажные параметры**

#### 3.1 Гамма-каротаж (ГК)
Для каждой литологии:
- **Базовое значение** (среднее)
- **Вариация** (стандартное отклонение)
- **Диапазон** (min-max)

**Типичные значения:**
```
Песок:           20-50 API
Глина:           60-120 API
Карбонаты:       10-40 API
Уголь:           10-30 API
Алевролит:       40-80 API
```

#### 3.2 Плотность (GGKP)
Для каждой литологии:
- **Матричная плотность** (плотность скелета породы)
- **Плотность флюида**
- **Наблюдаемая плотность** (среднее ± σ)

**Формула:**
```
ρ_bulk = φ × ρ_fluid + (1 - φ) × ρ_matrix
```

**Типичные матричные плотности:**
```
Песок:           2.65 г/см³
Глина:           2.50-2.60 г/см³
Карбонаты:       2.71 г/см³
Уголь:           1.30-1.50 г/см³
Алевролит:       2.60-2.65 г/см³
```

#### 3.3 Другие каротажи (если доступны)
- **НК (Нейтронный каротаж)**
- **АК (Акустический каротаж)**
- **ПС (Потенциал самопроизвольной поляризации)**
- **БК (Боковой каротаж)**

---

### 4️⃣ **Марковская модель переходов**

#### 4.1 Матрица переходов между литологиями
Вероятности перехода от одной породы к другой:

```python
def calculate_transition_matrix(lithology_sequence):
    """
    Вычисление матрицы переходов из последовательности литологий
    """
    unique_liths = sorted(set(lithology_sequence))
    n = len(unique_liths)
    lith_to_idx = {lith: i for i, lith in enumerate(unique_liths)}
    
    # Подсчет переходов
    transition_counts = np.zeros((n, n))
    for i in range(len(lithology_sequence) - 1):
        from_idx = lith_to_idx[lithology_sequence[i]]
        to_idx = lith_to_idx[lithology_sequence[i + 1]]
        transition_counts[from_idx, to_idx] += 1
    
    # Нормализация (получение вероятностей)
    transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    
    return transition_matrix, unique_liths
```

**Что анализировать:**
- Какие переходы наиболее вероятны?
- Есть ли циклы (трансгрессия/регрессия)?
- Какие переходы невозможны или редки?

---

### 5️⃣ **Геологические циклы**

#### 5.1 Циклы трансгрессии/регрессии
Если в разрезе есть цикличность:
- **Длина цикла** (в метрах)
- **Тип цикла** (трансгрессивный/регрессивный)
- **Амплитуда изменений**

**Признаки:**
- Трансгрессия: песок → алевролит → глина (укрупнение)
- Регрессия: глина → алевролит → песок (обмеление)

```python
# Анализ цикличности
def detect_cycles(df, window=100):
    """
    Определение цикличности через скользящее среднее
    """
    # Присвоим литологиям числовые значения (по крупности)
    lith_scale = {'sand': 1, 'siltstone': 2, 'shale': 3, 'carbonate': 1.5, 'coal': 0}
    df['Lith_numeric'] = df['Lithology'].map(lith_scale)
    
    # Скользящее среднее
    df['Lith_MA'] = df['Lith_numeric'].rolling(window=window).mean()
    
    # Поиск циклов (пики и впадины)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df['Lith_MA'].dropna())
    troughs, _ = find_peaks(-df['Lith_MA'].dropna())
    
    # Средняя длина цикла
    if len(peaks) > 1:
        cycle_length = np.mean(np.diff(peaks))
        print(f"Средняя длина цикла: {cycle_length:.1f} точек")
    
    return peaks, troughs
```

---

### 6️⃣ **Эффекты инверсии (смазывание границ)**

#### 6.1 Размер зоны инверсии
Расстояние, на которое "смазываются" границы слоев:
- **Вертикальное разрешение прибора**
- **Радиус исследования**

**Типичные значения:**
```
ГК:              30-50 см (медленный прибор)
GGKP:            15-30 см
НК:              30-50 см
АК:              60-100 см
```

#### 6.2 Степень влияния соседних слоев
```python
def estimate_invasion_effect(df, boundary_indices, window=5):
    """
    Оценка эффекта инверсии на границах слоев
    """
    invasion_effects = []
    
    for idx in boundary_indices:
        if idx > window and idx < len(df) - window:
            # Значение точно на границе
            boundary_value = df.iloc[idx]['Gamma']
            
            # Среднее значение в окне
            window_mean = df.iloc[idx-window:idx+window]['Gamma'].mean()
            
            # Разница показывает степень смазывания
            invasion_effects.append(abs(boundary_value - window_mean))
    
    return np.mean(invasion_effects)
```

---

### 7️⃣ **Шум и неопределенность**

#### 7.1 Уровень шума в каротаже
- **Signal-to-Noise Ratio (SNR)**
- **Стандартное отклонение шума**

```python
# Оценка шума (высокочастотная компонента)
def estimate_noise(signal, window=5):
    smooth = signal.rolling(window=window).mean()
    noise = signal - smooth
    noise_std = noise.std()
    return noise_std
```

#### 7.2 Выбросы и аномалии
- Процент аномальных значений
- Характер выбросов (случайные/систематические)

---

## 🔬 Пример анализа реальной скважины

### Шаг 1: Загрузка данных
```python
import pandas as pd
import numpy as np

# Загрузка данных скважины
df = pd.read_csv('real_well_data.csv')

# Необходимые колонки:
# - Depth (глубина)
# - Lithology (литология)
# - Gamma (ГК)
# - Rho (плотность)
# - Porosity (пористость, если есть)
# - Sw (водонасыщенность, если есть)
```

### Шаг 2: Базовая статистика по литологиям
```python
print("=" * 60)
print("СТАТИСТИКА ПО ЛИТОЛОГИЯМ")
print("=" * 60)

for lith in df['Lithology'].unique():
    lith_data = df[df['Lithology'] == lith]
    count = len(lith_data)
    percentage = (count / len(df)) * 100
    
    print(f"\n{lith}:")
    print(f"  Количество: {count} ({percentage:.1f}%)")
    
    if 'Porosity' in df.columns:
        phi_mean = lith_data['Porosity'].mean()
        phi_std = lith_data['Porosity'].std()
        print(f"  Пористость: {phi_mean:.3f} ± {phi_std:.3f}")
    
    if 'Sw' in df.columns:
        sw_mean = lith_data['Sw'].mean()
        sw_std = lith_data['Sw'].std()
        print(f"  Sw: {sw_mean:.3f} ± {sw_std:.3f}")
    
    gamma_mean = lith_data['Gamma'].mean()
    gamma_std = lith_data['Gamma'].std()
    print(f"  ГК: {gamma_mean:.1f} ± {gamma_std:.1f} API")
    
    rho_mean = lith_data['Rho'].mean()
    rho_std = lith_data['Rho'].std()
    print(f"  Плотность: {rho_mean:.2f} ± {rho_std:.2f} г/см³")
```

### Шаг 3: Анализ длин серий
```python
print("\n" + "=" * 60)
print("ДЛИНЫ ЛИТОЛОГИЧЕСКИХ СЕРИЙ")
print("=" * 60)

series_stats = analyze_series_lengths(df['Lithology'])

for lith, lengths in series_stats.items():
    print(f"\n{lith}:")
    print(f"  Min: {min(lengths)} точек")
    print(f"  Max: {max(lengths)} точек")
    print(f"  Mean: {np.mean(lengths):.1f} точек")
    print(f"  Median: {np.median(lengths):.1f} точек")
```

### Шаг 4: Матрица переходов
```python
print("\n" + "=" * 60)
print("МАТРИЦА ПЕРЕХОДОВ МЕЖДУ ЛИТОЛОГИЯМИ")
print("=" * 60)

trans_matrix, liths = calculate_transition_matrix(df['Lithology'].tolist())

print("\nВероятности переходов:")
print(pd.DataFrame(trans_matrix, index=liths, columns=liths).round(3))
```

### Шаг 5: Генерация конфигурационного файла
```python
# Создание конфига для генератора
config = {
    'lithology_freq_range': {
        lith: (freq * 0.8, freq * 1.2)  # ±20% от наблюдаемой частоты
        for lith, freq in lithology_frequencies.items()
    },
    'series_length_range': {
        lith: (min(lengths), max(lengths))
        for lith, lengths in series_stats.items()
    },
    'lithology_properties': {
        lith: {
            'porosity': (phi_mean, phi_std),
            'sw': (sw_mean, sw_std),
            'gamma_base': gamma_mean,
            'matrix_density': estimate_matrix_density(lith_data)
        }
        for lith in df['Lithology'].unique()
    },
    'transition_matrix': trans_matrix.tolist()
}

# Сохранение в JSON
import json
with open('well_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ Конфигурация сохранена в well_config.json")
```

---

## 📊 Минимальный набор данных

Если полная информация недоступна, минимум необходимый для калибровки:

### Обязательно:
1. ✅ **Литологическая колонка** (последовательность пород)
2. ✅ **ГК (Гамма-каротаж)** - самый распространенный
3. ✅ **Глубина**

### Желательно:
4. 🟡 **GGKP (Плотность)**
5. 🟡 **НК (Нейтронный каротаж)**
6. 🟡 **Пористость** (из интерпретации или керна)

### Дополнительно:
7. 🔵 **Водонасыщенность**
8. 🔵 **АК (Акустический каротаж)**
9. 🔵 **Данные керна** (для калибровки)

---

## 🎯 Рекомендации по использованию

### 1. Количество скважин для калибровки
- **Минимум**: 1-2 скважины (базовая статистика)
- **Оптимум**: 5-10 скважин (надежная статистика)
- **Идеал**: 20+ скважин (учет вариабельности)

### 2. Зоны калибровки
Если месторождение большое:
- Калибровка по **геологическим зонам**
- Отдельные параметры для **разных пластов**
- Учет **фациальных изменений**

### 3. Валидация
После калибровки:
1. Сгенерировать тестовую скважину
2. Сравнить статистику с реальными данными
3. Проверить визуальное сходство
4. Скорректировать параметры при необходимости

---

## 📝 Чек-лист параметров для калибровки

### Литология
- [ ] Частоты встречаемости (min-max)
- [ ] Длины серий (min-mean-max)
- [ ] Матрица переходов
- [ ] Наличие циклов (да/нет, длина)

### Физические свойства
- [ ] Пористость (μ, σ) для каждой литологии
- [ ] Водонасыщенность (μ, σ) для каждой литологии
- [ ] Корреляции между параметрами
- [ ] Матричные плотности

### Каротаж
- [ ] ГК: базовые значения и вариации
- [ ] GGKP: базовые значения и вариации
- [ ] НК (если есть): базовые значения
- [ ] АК (если есть): базовые значения

### Эффекты
- [ ] Размер зоны инверсии (см/м)
- [ ] Степень смазывания границ
- [ ] Уровень шума
- [ ] Характеристики выбросов

---

## 🔧 Инструменты для анализа

Создайте скрипт `analyze_real_well.py` для автоматического извлечения всех параметров из реальных данных.

См. также:
- `calibration_script.py` - готовый скрипт анализа
- `DEPLOYMENT.md` - инструкции по деплою
- `README.md` - общая информация

---

**🛢️ Генератор синтетических скважин** | Калибровка на реальных данных
