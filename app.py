import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Настройка страницы
st.set_page_config(
    page_title="Генератор синтетических скважин",
    page_icon="🛢️",
    layout="wide"
)

# Заголовок приложения
st.title("🛢️ Генератор синтетических скважин")
st.markdown("Приложение для генерации синтетических каротажных данных с использованием Марковских цепей и циклов трансгрессии/регрессии")

# Боковая панель с параметрами
st.sidebar.header("Параметры модели")

# Основные параметры
n_depth = st.sidebar.slider("Количество точек по глубине", 100, 2000, 1000, 50)
depth_start = st.sidebar.number_input("Начальная глубина (м)", 0, 5000, 1000)
depth_end = st.sidebar.number_input("Конечная глубина (м)", 1000, 10000, 2000)

# Параметры циклов
cycle_length = st.sidebar.slider("Длина цикла трансгрессии/регрессии", 50, 500, 200, 25)

# Параметры инверсии
invasion_window = st.sidebar.slider("Окно инверсии", 3, 15, 7, 1)
invasion_strength = st.sidebar.slider("Сила инверсии", 0.0, 1.0, 0.6, 0.1)

# Частоты встречаемости литологий
st.sidebar.subheader("Частота встречаемости литологий")
lithology_freq_range = {}
lithology_states = ['sand', 'shale', 'carbonate_sand', 'coal', 'siltstone']
lithology_names = ['Песок', 'Глина', 'Карбонатный песок', 'Уголь', 'Алевролит']

for i, (lith, name) in enumerate(zip(lithology_states, lithology_names)):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_freq = st.number_input(f"{name} мин", 0.0, 1.0, 
                                 [0.01, 0.35, 0.05, 0.01, 0.05][i], 0.01, key=f"min_{lith}")
    with col2:
        max_freq = st.number_input(f"{name} макс", 0.0, 1.0, 
                                 [0.15, 0.9, 0.12, 0.08, 0.15][i], 0.01, key=f"max_{lith}")
    lithology_freq_range[lith] = (min_freq, max_freq)

# Диапазоны длины серий
st.sidebar.subheader("Длина серий литологий")
series_length_range = {}
for i, (lith, name) in enumerate(zip(lithology_states, lithology_names)):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_len = st.number_input(f"{name} мин", 1, 50, 
                                [2, 1, 2, 1, 2][i], 1, key=f"min_len_{lith}")
    with col2:
        max_len = st.number_input(f"{name} макс", 1, 200, 
                                [10, 155, 6, 4, 20][i], 1, key=f"max_len_{lith}")
    series_length_range[lith] = (min_len, max_len)

# Функции генерации
def sample_initial_prob(freq_range):
    """Генерация начальных вероятностей"""
    freqs = [np.random.uniform(low, high) for (low, high) in freq_range.values()]
    freqs = np.array(freqs)
    freqs /= freqs.sum()
    return freqs

def generate_lithology_markov_cycles(n, states, trans_matrix, initial_prob, series_range, cycle_len=200):
    """Генерация литологии с циклами трансгрессии/регрессии"""
    lithology = []
    current_lith = np.random.choice(states, p=initial_prob)
    lithology.append(current_lith)

    for i in range(1, n):
        cycle_phase = (i // cycle_len) % 2  # 0 = трансгрессия, 1 = регрессия

        if cycle_phase == 0:  # Трансгрессия: sand→siltstone→shale
            bias_matrix = np.array([
                [0.4, 0.2, 0.05, 0.05, 0.3],
                [0.2, 0.4, 0.1, 0.05, 0.25],
                [0.1, 0.2, 0.5, 0.05, 0.15],
                [0.1, 0.1, 0.05, 0.7, 0.05],
                [0.15, 0.25, 0.2, 0.05, 0.35]
            ])
        else:  # Регрессия: shale→siltstone→sand
            bias_matrix = np.array([
                [0.6, 0.1, 0.1, 0.05, 0.15],
                [0.3, 0.3, 0.05, 0.05, 0.3],
                [0.3, 0.1, 0.4, 0.05, 0.15],
                [0.1, 0.1, 0.1, 0.6, 0.1],
                [0.3, 0.2, 0.1, 0.05, 0.35]
            ])

        prev_idx = states.index(current_lith)
        probs = 0.5 * trans_matrix[prev_idx] + 0.5 * bias_matrix[prev_idx]
        probs /= probs.sum()
        current_lith = np.random.choice(states, p=probs)
        lithology.append(current_lith)

    # Растягиваем серии
    lith_expanded = []
    for lith in lithology:
        min_len, max_len = series_range[lith]
        run_len = np.random.randint(min_len, max_len + 1)
        lith_expanded.extend([lith] * run_len)

    return lith_expanded[:n]

def apply_invasion_effect(curve, window=5, alpha=0.5):
    """Применение эффекта инверсии"""
    curve = np.array(curve)
    kernel = np.ones(window) / window
    smooth = np.convolve(curve, kernel, mode="same")
    return alpha*smooth + (1-alpha)*curve

# Базовая матрица переходов
transition_matrix = np.array([
    [0.6, 0.2, 0.1, 0.05, 0.05],   # sand
    [0.3, 0.5, 0.1, 0.05, 0.05],   # shale
    [0.2, 0.1, 0.6, 0.05, 0.05],   # carbonate_sand
    [0.1, 0.1, 0.1, 0.6, 0.1],     # coal
    [0.2, 0.2, 0.2, 0.05, 0.35]    # siltstone
])

# Кнопка генерации
if st.sidebar.button("🔄 Сгенерировать скважину", type="primary"):
    with st.spinner("Генерируем синтетическую скважину..."):
        # Генерация начальных вероятностей
        initial_prob = sample_initial_prob(lithology_freq_range)
        
        # Генерация литологии
        lithology = generate_lithology_markov_cycles(
            n_depth, lithology_states, transition_matrix, initial_prob, 
            series_length_range, cycle_len=cycle_length
        )
        
        # Генерация глубин
        depths = np.linspace(depth_start, depth_end, n_depth)
        
        # Генерация пористости и насыщения
        phi, sw = [], []
        for lith in lithology:
            if lith == 'sand':
                phi.append(np.random.normal(0.25, 0.03))
                sw.append(np.random.normal(0.3, 0.05))
            elif lith == 'shale':
                phi.append(np.random.normal(0.15, 0.02))
                sw.append(np.random.normal(0.8, 0.05))
            elif lith == 'carbonate_sand':
                phi.append(np.random.normal(0.20, 0.02))
                sw.append(np.random.normal(0.4, 0.05))
            elif lith == 'coal':
                phi.append(np.random.normal(0.35, 0.04))
                sw.append(np.random.normal(0.2, 0.05))
            elif lith == 'siltstone':
                phi.append(np.random.normal(0.18, 0.02))
                sw.append(np.random.normal(0.5, 0.05))
        
        # Параметры каротажа
        matrix_density = {'sand':2.65, 'shale':2.55, 'carbonate_sand':2.71, 'coal':1.4, 'siltstone':2.62}
        rho_fluid = 1.0
        gamma_base = {'sand':40, 'shale':70, 'carbonate_sand':35, 'coal':30, 'siltstone':60}
        
        # Генерация каротажных кривых
        gamma_log, rho_log = [], []
        for i, lith in enumerate(lithology):
            gamma_log.append(np.random.normal(gamma_base[lith], 3))
            rho = phi[i]*rho_fluid + (1-phi[i])*matrix_density[lith]
            rho += np.random.normal(0, 0.02)
            rho_log.append(rho)
        
        # Применение эффекта инверсии
        gamma_inv = apply_invasion_effect(gamma_log, window=invasion_window, alpha=invasion_strength)
        rho_inv = apply_invasion_effect(rho_log, window=invasion_window, alpha=invasion_strength)
        
        # Создание DataFrame
        df = pd.DataFrame({
            'Depth': depths,
            'Lithology': lithology,
            'Porosity': phi,
            'Sw': sw,
            'Gamma_clean': gamma_log,
            'Gamma_inv': gamma_inv,
            'Rho_clean': rho_log,
            'Rho_inv': rho_inv
        })
        
        # Сохранение в session state
        st.session_state['well_data'] = df
        st.session_state['well_generated'] = True

# Отображение результатов
if st.session_state.get('well_generated', False):
    df = st.session_state['well_data']
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Глубина интервала", f"{depth_start}-{depth_end} м")
    with col2:
        st.metric("Количество точек", len(df))
    with col3:
        st.metric("Средняя пористость", f"{df['Porosity'].mean():.3f}")
    with col4:
        st.metric("Среднее насыщение", f"{df['Sw'].mean():.3f}")
    
    # Распределение литологий
    st.subheader("📊 Распределение литологий")
    lith_counts = df['Lithology'].value_counts()
    lith_percentages = (lith_counts / len(df) * 100).round(1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(lith_percentages)
    with col2:
        for lith, count in lith_counts.items():
            percentage = lith_percentages[lith]
            st.write(f"**{lith}**: {count} точек ({percentage}%)")
    
    # Визуализация каротажных кривых
    st.subheader("📈 Каротажные кривые")
    
    # Настройка matplotlib для русского языка
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    
    # ГК
    axs[0].plot(df['Gamma_clean'], df['Depth'], label='Чистый ГК', color='green', alpha=0.5, linewidth=0.8)
    axs[0].plot(df['Gamma_inv'], df['Depth'], label='ГК с инверсией', color='black', linewidth=1)
    axs[0].set_title('Гамма-каротаж')
    axs[0].set_xlabel('ГК (API)')
    axs[0].set_ylabel('Глубина (м)')
    axs[0].invert_yaxis()
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Плотность
    axs[1].plot(df['Rho_clean'], df['Depth'], label='Чистая плотность', color='blue', alpha=0.5, linewidth=0.8)
    axs[1].plot(df['Rho_inv'], df['Depth'], label='Плотность с инверсией', color='red', linewidth=1)
    axs[1].set_title('ГГКП (Density)')
    axs[1].set_xlabel('Плотность (г/см³)')
    axs[1].invert_yaxis()
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Пористость
    axs[2].plot(df['Porosity'], df['Depth'], color='orange', linewidth=1)
    axs[2].set_title('Пористость')
    axs[2].set_xlabel('Пористость (доли)')
    axs[2].invert_yaxis()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Кроссплоты
    st.subheader("🔍 Кроссплоты")
    
    palette_lith = {
        'sand': 'yellow', 'shale': 'brown', 'carbonate_sand': 'orange', 
        'coal': 'black', 'siltstone': 'gray'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x='Gamma_inv', y='Rho_inv', hue='Lithology', 
                       palette=palette_lith, alpha=0.7, ax=ax)
        ax.set_title('Crossplot Gamma vs Density по литологии')
        ax.set_xlabel('ГК (API)')
        ax.set_ylabel('Плотность (г/см³)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df['Gamma_inv'], df['Rho_inv'], c=df['Porosity'], 
                       cmap='viridis', alpha=0.7)
        ax.set_title('Crossplot Gamma vs Density по пористости')
        ax.set_xlabel('ГК (API)')
        ax.set_ylabel('Плотность (г/см³)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Пористость')
        st.pyplot(fig)
    
    # Дополнительный кроссплот
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Rho_inv', y='Porosity', hue='Lithology', 
                   palette=palette_lith, alpha=0.7, ax=ax)
    ax.set_title('Crossplot Density vs Porosity по литологии')
    ax.set_xlabel('Плотность (г/см³)')
    ax.set_ylabel('Пористость (доли)')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Таблица данных
    st.subheader("📋 Данные скважины")
    
    # Фильтрация данных
    col1, col2 = st.columns(2)
    with col1:
        show_clean = st.checkbox("Показать чистые кривые", value=False)
    with col2:
        depth_filter = st.slider("Фильтр по глубине", 
                                float(df['Depth'].min()), 
                                float(df['Depth'].max()), 
                                (float(df['Depth'].min()), float(df['Depth'].max())))
    
    # Применение фильтров
    df_filtered = df[(df['Depth'] >= depth_filter[0]) & (df['Depth'] <= depth_filter[1])]
    
    if not show_clean:
        df_display = df_filtered.drop(['Gamma_clean', 'Rho_clean'], axis=1)
    else:
        df_display = df_filtered
    
    st.dataframe(df_display, use_container_width=True)
    
    # Экспорт данных
    st.subheader("💾 Экспорт данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Скачать CSV",
            data=csv,
            file_name=f"synthetic_well_{depth_start}_{depth_end}m.csv",
            mime="text/csv"
        )
    
    with col2:
        # Создание Excel файла
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Well_Data', index=False)
            
            # Добавление листа со статистикой
            stats_df = pd.DataFrame({
                'Parameter': ['Total Points', 'Depth Range (m)', 'Avg Porosity', 'Avg Sw', 'Avg Gamma', 'Avg Density'],
                'Value': [
                    len(df),
                    f"{df['Depth'].min():.1f} - {df['Depth'].max():.1f}",
                    f"{df['Porosity'].mean():.3f}",
                    f"{df['Sw'].mean():.3f}",
                    f"{df['Gamma_inv'].mean():.1f}",
                    f"{df['Rho_inv'].mean():.2f}"
                ]
            })
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        st.download_button(
            label="📥 Скачать Excel",
            data=output.getvalue(),
            file_name=f"synthetic_well_{depth_start}_{depth_end}m.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info("👈 Настройте параметры в боковой панели и нажмите 'Сгенерировать скважину' для создания синтетических данных")

# Информация о модели
with st.expander("ℹ️ О модели"):
    st.markdown("""
    ### Описание модели генерации синтетических скважин
    
    **Основные компоненты:**
    
    1. **Марковская цепь** - моделирует переходы между литологиями
    2. **Циклы трансгрессии/регрессии** - имитирует геологические процессы
    3. **Эффект инверсии** - смазывание границ слоев (реалистичность)
    
    **Литологические типы:**
    - **Песок** - высокая пористость, низкое насыщение
    - **Глина** - низкая пористость, высокое насыщение  
    - **Карбонатный песок** - средние значения
    - **Уголь** - очень высокая пористость, низкое насыщение
    - **Алевролит** - промежуточные свойства
    
    **Каротажные кривые:**
    - **ГК (Гамма-каротаж)** - радиоактивность пород
    - **ГГКП (Гамма-гамма каротаж по плотности)** - плотность пород
    - **Пористость** - объем пустот в породе
    - **Насыщение** - доля воды в порах
    """)

# Футер
st.markdown("---")
st.markdown("🛢️ **Генератор синтетических скважин** | Создано с использованием Streamlit")
