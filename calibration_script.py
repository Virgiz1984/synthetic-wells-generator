"""
Скрипт для калибровки генератора синтетических скважин на реальных данных

Использование:
    python calibration_script.py --input real_well_data.csv --output well_config.json

Формат входных данных (CSV):
    - Depth: глубина (м)
    - Lithology: литология (текст)
    - Gamma: гамма-каротаж (API)
    - Rho: плотность (г/см³)
    - Porosity: пористость (опционально)
    - Sw: водонасыщенность (опционально)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from collections import defaultdict


class WellCalibrator:
    """Класс для калибровки параметров генератора на реальных данных"""
    
    def __init__(self, df):
        """
        Инициализация калибратора
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame с данными скважины
        """
        self.df = df
        self.lithologies = df['Lithology'].unique()
        self.config = {}
        
    def analyze_lithology_frequencies(self):
        """Анализ частот встречаемости литологий"""
        print("\n" + "="*70)
        print("1️⃣  ЧАСТОТЫ ВСТРЕЧАЕМОСТИ ЛИТОЛОГИЙ")
        print("="*70)
        
        counts = self.df['Lithology'].value_counts()
        frequencies = counts / len(self.df)
        
        freq_range = {}
        for lith in self.lithologies:
            freq = frequencies[lith]
            # Диапазон ±20% от наблюдаемой частоты
            freq_range[lith] = (max(0.01, freq * 0.8), min(1.0, freq * 1.2))
            print(f"{lith:20s}: {counts[lith]:5d} точек ({freq*100:5.2f}%) → диапазон: ({freq_range[lith][0]:.3f}, {freq_range[lith][1]:.3f})")
        
        self.config['lithology_freq_range'] = freq_range
        return freq_range
    
    def analyze_series_lengths(self):
        """Анализ длин литологических серий"""
        print("\n" + "="*70)
        print("2️⃣  ДЛИНЫ ЛИТОЛОГИЧЕСКИХ СЕРИЙ")
        print("="*70)
        
        series_lengths = defaultdict(list)
        
        current_lith = self.df['Lithology'].iloc[0]
        current_length = 1
        
        for lith in self.df['Lithology'].iloc[1:]:
            if lith == current_lith:
                current_length += 1
            else:
                series_lengths[current_lith].append(current_length)
                current_lith = lith
                current_length = 1
        
        # Добавим последнюю серию
        series_lengths[current_lith].append(current_length)
        
        series_range = {}
        for lith in self.lithologies:
            if lith in series_lengths and len(series_lengths[lith]) > 0:
                lengths = series_lengths[lith]
                min_len = int(min(lengths))
                max_len = int(max(lengths))
                mean_len = np.mean(lengths)
                median_len = np.median(lengths)
                
                series_range[lith] = (min_len, max_len)
                
                print(f"{lith:20s}: min={min_len:4d}, max={max_len:4d}, mean={mean_len:6.1f}, median={median_len:6.1f}")
        
        self.config['series_length_range'] = series_range
        return series_range, series_lengths
    
    def analyze_physical_properties(self):
        """Анализ физических свойств пород"""
        print("\n" + "="*70)
        print("3️⃣  ФИЗИЧЕСКИЕ СВОЙСТВА ПОРОД")
        print("="*70)
        
        properties = {}
        
        for lith in self.lithologies:
            lith_data = self.df[self.df['Lithology'] == lith]
            props = {}
            
            print(f"\n{lith}:")
            
            # Пористость
            if 'Porosity' in self.df.columns:
                phi_mean = lith_data['Porosity'].mean()
                phi_std = lith_data['Porosity'].std()
                props['porosity'] = (float(phi_mean), float(phi_std))
                print(f"  Пористость:       {phi_mean:.3f} ± {phi_std:.3f}")
            
            # Водонасыщенность
            if 'Sw' in self.df.columns:
                sw_mean = lith_data['Sw'].mean()
                sw_std = lith_data['Sw'].std()
                props['sw'] = (float(sw_mean), float(sw_std))
                print(f"  Sw:                {sw_mean:.3f} ± {sw_std:.3f}")
            
            # Гамма-каротаж
            if 'Gamma' in self.df.columns:
                gamma_mean = lith_data['Gamma'].mean()
                gamma_std = lith_data['Gamma'].std()
                props['gamma_base'] = float(gamma_mean)
                props['gamma_std'] = float(gamma_std)
                print(f"  ГК:                {gamma_mean:.1f} ± {gamma_std:.1f} API")
            
            # Плотность
            if 'Rho' in self.df.columns:
                rho_mean = lith_data['Rho'].mean()
                rho_std = lith_data['Rho'].std()
                props['observed_density'] = (float(rho_mean), float(rho_std))
                print(f"  Плотность:         {rho_mean:.2f} ± {rho_std:.2f} г/см³")
                
                # Оценка матричной плотности
                if 'Porosity' in self.df.columns:
                    phi_avg = lith_data['Porosity'].mean()
                    rho_fluid = 1.0  # предполагаем воду
                    # ρ_bulk = φ × ρ_fluid + (1 - φ) × ρ_matrix
                    # ρ_matrix = (ρ_bulk - φ × ρ_fluid) / (1 - φ)
                    if phi_avg < 0.99:
                        rho_matrix = (rho_mean - phi_avg * rho_fluid) / (1 - phi_avg)
                        props['matrix_density'] = float(rho_matrix)
                        print(f"  Матр. плотность:   {rho_matrix:.2f} г/см³ (расчетная)")
            
            # Цвета для визуализации (можно настроить)
            color_map = {
                'sand': 'yellow', 'shale': 'brown', 'carbonate': 'orange',
                'carbonate_sand': 'orange', 'coal': 'black', 'siltstone': 'gray',
                'limestone': 'lightblue', 'sandstone': 'gold'
            }
            props['color'] = color_map.get(lith.lower(), 'gray')
            
            properties[lith] = props
        
        self.config['lithology_properties'] = properties
        return properties
    
    def calculate_transition_matrix(self):
        """Расчет матрицы переходов между литологиями"""
        print("\n" + "="*70)
        print("4️⃣  МАТРИЦА ПЕРЕХОДОВ МЕЖДУ ЛИТОЛОГИЯМИ")
        print("="*70)
        
        lithology_sequence = self.df['Lithology'].tolist()
        unique_liths = sorted(set(lithology_sequence))
        n = len(unique_liths)
        lith_to_idx = {lith: i for i, lith in enumerate(unique_liths)}
        
        # Подсчет переходов
        transition_counts = np.zeros((n, n))
        for i in range(len(lithology_sequence) - 1):
            from_idx = lith_to_idx[lithology_sequence[i]]
            to_idx = lith_to_idx[lithology_sequence[i + 1]]
            transition_counts[from_idx, to_idx] += 1
        
        # Нормализация
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # избегаем деления на ноль
        transition_matrix = transition_counts / row_sums
        
        # Вывод матрицы
        print("\nВероятности переходов:")
        df_trans = pd.DataFrame(transition_matrix, index=unique_liths, columns=unique_liths)
        print(df_trans.round(3))
        
        self.config['transition_matrix'] = transition_matrix.tolist()
        self.config['lithology_states'] = unique_liths
        
        return transition_matrix, unique_liths
    
    def analyze_noise_and_smoothing(self):
        """Анализ уровня шума и эффектов сглаживания"""
        print("\n" + "="*70)
        print("5️⃣  ШУМЫ И ЭФФЕКТЫ СГЛАЖИВАНИЯ")
        print("="*70)
        
        if 'Gamma' in self.df.columns:
            # Оценка шума в ГК
            window = 5
            smooth = self.df['Gamma'].rolling(window=window, center=True).mean()
            noise = self.df['Gamma'] - smooth
            noise_std = noise.std()
            
            print(f"\nГамма-каротаж:")
            print(f"  Уровень шума (σ):  {noise_std:.2f} API")
            
            # SNR (Signal-to-Noise Ratio)
            signal_std = self.df['Gamma'].std()
            snr = signal_std / noise_std if noise_std > 0 else float('inf')
            print(f"  SNR:               {snr:.1f}")
            
            self.config['gamma_noise_std'] = float(noise_std)
        
        if 'Rho' in self.df.columns:
            # Оценка шума в плотности
            window = 5
            smooth = self.df['Rho'].rolling(window=window, center=True).mean()
            noise = self.df['Rho'] - smooth
            noise_std = noise.std()
            
            print(f"\nПлотность:")
            print(f"  Уровень шума (σ):  {noise_std:.3f} г/см³")
            
            self.config['rho_noise_std'] = float(noise_std)
        
        # Рекомендации по окну инверсии
        print(f"\nРекомендации:")
        print(f"  Окно инверсии:     5-10 точек")
        print(f"  Сила инверсии:     0.5-0.7")
        
        return self.config
    
    def analyze_correlations(self):
        """Анализ корреляций между параметрами"""
        print("\n" + "="*70)
        print("6️⃣  КОРРЕЛЯЦИИ МЕЖДУ ПАРАМЕТРАМИ")
        print("="*70)
        
        corr_cols = []
        if 'Porosity' in self.df.columns:
            corr_cols.append('Porosity')
        if 'Sw' in self.df.columns:
            corr_cols.append('Sw')
        if 'Gamma' in self.df.columns:
            corr_cols.append('Gamma')
        if 'Rho' in self.df.columns:
            corr_cols.append('Rho')
        
        if len(corr_cols) >= 2:
            print("\nОбщая корреляционная матрица:")
            corr_matrix = self.df[corr_cols].corr()
            print(corr_matrix.round(3))
            
            # Корреляции по литологиям
            print("\nКорреляции по литологиям:")
            for lith in self.lithologies:
                lith_data = self.df[self.df['Lithology'] == lith]
                if len(lith_data) > 10:  # минимум 10 точек
                    print(f"\n{lith}:")
                    lith_corr = lith_data[corr_cols].corr()
                    print(lith_corr.round(3))
        
        return corr_matrix if len(corr_cols) >= 2 else None
    
    def generate_visualizations(self, output_prefix='calibration'):
        """Генерация визуализаций для анализа"""
        print("\n" + "="*70)
        print("7️⃣  ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИЙ")
        print("="*70)
        
        # 1. Распределение литологий
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        lith_counts = self.df['Lithology'].value_counts()
        ax1.bar(range(len(lith_counts)), lith_counts.values, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(lith_counts)))
        ax1.set_xticklabels(lith_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Количество точек')
        ax1.set_title('Распределение литологий')
        ax1.grid(alpha=0.3, axis='y')
        
        ax2.pie(lith_counts.values, labels=lith_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Процентное соотношение')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_lithology_distribution.png', dpi=150)
        print(f"✅ Сохранено: {output_prefix}_lithology_distribution.png")
        plt.close()
        
        # 2. Каротажные кривые
        if 'Gamma' in self.df.columns and 'Depth' in self.df.columns:
            fig, axes = plt.subplots(1, 3, figsize=(14, 10), sharey=True)
            
            if 'Gamma' in self.df.columns:
                axes[0].plot(self.df['Gamma'], self.df['Depth'], 'g-', linewidth=0.5)
                axes[0].set_xlabel('ГК (API)')
                axes[0].set_ylabel('Глубина (м)')
                axes[0].set_title('Гамма-каротаж')
                axes[0].invert_yaxis()
                axes[0].grid(alpha=0.3)
            
            if 'Rho' in self.df.columns:
                axes[1].plot(self.df['Rho'], self.df['Depth'], 'b-', linewidth=0.5)
                axes[1].set_xlabel('Плотность (г/см³)')
                axes[1].set_title('Плотность')
                axes[1].grid(alpha=0.3)
            
            if 'Porosity' in self.df.columns:
                axes[2].plot(self.df['Porosity'], self.df['Depth'], 'orange', linewidth=0.5)
                axes[2].set_xlabel('Пористость')
                axes[2].set_title('Пористость')
                axes[2].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_prefix}_log_curves.png', dpi=150)
            print(f"✅ Сохранено: {output_prefix}_log_curves.png")
            plt.close()
        
        # 3. Кроссплоты
        if 'Gamma' in self.df.columns and 'Rho' in self.df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            for lith in self.lithologies:
                lith_data = self.df[self.df['Lithology'] == lith]
                ax.scatter(lith_data['Gamma'], lith_data['Rho'], 
                          label=lith, alpha=0.6, s=20)
            
            ax.set_xlabel('ГК (API)')
            ax.set_ylabel('Плотность (г/см³)')
            ax.set_title('Кроссплот Gamma vs Density')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_prefix}_crossplot.png', dpi=150)
            print(f"✅ Сохранено: {output_prefix}_crossplot.png")
            plt.close()
        
        print("\n✅ Все визуализации созданы!")
    
    def save_config(self, output_file='well_config.json'):
        """Сохранение конфигурации в JSON"""
        print("\n" + "="*70)
        print("💾 СОХРАНЕНИЕ КОНФИГУРАЦИИ")
        print("="*70)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Конфигурация сохранена в: {output_file}")
        print(f"📊 Размер конфига: {len(json.dumps(self.config))} байт")
        
        return output_file
    
    def run_full_analysis(self, output_prefix='calibration'):
        """Запуск полного анализа"""
        print("\n" + "🔬"*35)
        print("КАЛИБРОВКА ГЕНЕРАТОРА СИНТЕТИЧЕСКИХ СКВАЖИН")
        print("🔬"*35)
        print(f"\nИсходные данные: {len(self.df)} точек, {len(self.lithologies)} литологий")
        
        # Выполнение всех анализов
        self.analyze_lithology_frequencies()
        self.analyze_series_lengths()
        self.analyze_physical_properties()
        self.calculate_transition_matrix()
        self.analyze_noise_and_smoothing()
        self.analyze_correlations()
        
        # Визуализации
        self.generate_visualizations(output_prefix)
        
        # Сохранение конфига
        config_file = f'{output_prefix}_config.json'
        self.save_config(config_file)
        
        print("\n" + "="*70)
        print("✅ КАЛИБРОВКА ЗАВЕРШЕНА!")
        print("="*70)
        print(f"\nФайлы результатов:")
        print(f"  📄 {config_file}")
        print(f"  🖼️ {output_prefix}_lithology_distribution.png")
        print(f"  🖼️ {output_prefix}_log_curves.png")
        print(f"  🖼️ {output_prefix}_crossplot.png")
        
        return self.config


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Калибровка генератора синтетических скважин на реальных данных'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Путь к CSV файлу с данными скважины')
    parser.add_argument('--output', '-o', type=str, default='calibration',
                       help='Префикс для выходных файлов (default: calibration)')
    
    args = parser.parse_args()
    
    # Загрузка данных
    print(f"📂 Загрузка данных из: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"❌ Ошибка загрузки файла: {e}")
        return
    
    # Проверка обязательных колонок
    required_cols = ['Depth', 'Lithology']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Отсутствуют обязательные колонки: {missing_cols}")
        print(f"Доступные колонки: {list(df.columns)}")
        return
    
    print(f"✅ Данные загружены: {df.shape[0]} строк × {df.shape[1]} столбцов")
    print(f"Колонки: {', '.join(df.columns)}")
    
    # Создание калибратора и запуск анализа
    calibrator = WellCalibrator(df)
    config = calibrator.run_full_analysis(args.output)
    
    print(f"\n🎯 Готово! Используйте {args.output}_config.json для генерации синтетических скважин.")


if __name__ == '__main__':
    main()

