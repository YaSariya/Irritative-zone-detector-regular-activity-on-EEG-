import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import pywt
from scipy.signal import butter, filtfilt, hilbert, periodogram, find_peaks
import pandas as pd
from datetime import datetime
import os
import io
import base64
from matplotlib.colors import LinearSegmentedColormap

# Настройка страницы
st.set_page_config(
    page_title="Поиск ритмичной активности",
    #page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EEGRhythmicActivityDetector:
    def __init__(self, edf_path=None, file_buffer=None):
        self.edf_path = edf_path
        self.file_buffer = file_buffer
        self.signal_data = None
        self.signal_fs = None
        self.channel_names = None
        self.detected_intervals = []
        self.all_channel_results = {}
        self.file_info = {}
        
    def load_edf_data(self, load_all_channels=True, verbose=False):
        try:
            if self.file_buffer:
                # Сохраняем временный файл из buffer
                with open("temp_edf_file.edf", "wb") as f:
                    f.write(self.file_buffer.getvalue())
                edf_path = "temp_edf_file.edf"
            else:
                edf_path = self.edf_path
                
            with pyedflib.EdfReader(edf_path) as edf:
                self.channel_names = edf.getSignalLabels()
                n_channels = edf.signals_in_file
                
                # Сохраняем информацию о файле
                self.file_info = {
                    'file_name': os.path.basename(edf_path),
                    'duration': edf.getFileDuration(),
                    'start_time': edf.getStartdatetime(),
                    'channels_count': n_channels
                }
                
                if load_all_channels:
                    # Загрузка всех каналов
                    self.multi_channel_data = []
                    for i in range(n_channels):
                        fs = int(edf.getSampleFrequency(i))
                        data = edf.readSignal(i)
                        self.multi_channel_data.append({
                            'name': self.channel_names[i],
                            'data': data,
                            'fs': fs
                        })
                    self.signal_fs = self.multi_channel_data[0]['fs']
                    self.signal_data = self.multi_channel_data[0]['data']
                
                # Получение временной оси
                if hasattr(self, 'signal_data'):
                    self.times = np.arange(len(self.signal_data)) / self.signal_fs
                
                # Удаляем временный файл если использовали buffer
                if self.file_buffer and os.path.exists("temp_edf_file.edf"):
                    os.remove("temp_edf_file.edf")
                    
        except Exception as e:
            st.error(f"Ошибка загрузки EDF файла: {e}")
            raise
    
    def preprocess_signal(self, signal, fs, low_freq=1, high_freq=40, notch_freq=50):
        # Полосовой фильтр Баттерворта
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if low >= 1 or high >= 1:
            st.warning("Предупреждение: частота среза превышает Найквиста!")
            return signal.copy()
        
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal
    
    def find_rhythmic_patterns(self, filtered_signal, fs, target_freq=25, 
                              min_waves=6, max_freq_variation=0.4, max_amp_variation=0.7):
        """
        Улучшенный алгоритм поиска ритмических паттернов
        с учетом естественной вариабельности волн
        """
        try:
            # Нормализуем сигнал для лучшего детектирования
            signal_normalized = filtered_signal / (np.max(np.abs(filtered_signal)) + 1e-8)
            
            # Находим все пики (положительные и отрицательные)
            positive_peaks, _ = find_peaks(signal_normalized, 
                                         height=0.2, 
                                         distance=int(fs/target_freq*0.6))
            negative_peaks, _ = find_peaks(-signal_normalized, 
                                         height=0.2, 
                                         distance=int(fs/target_freq*0.6))
            
            # Объединяем все экстремумы и сортируем
            all_extremas = np.sort(np.concatenate([positive_peaks, negative_peaks]))
            
            if len(all_extremas) < min_waves + 1:
                return []
            
            rhythmic_segments = []
            current_segment = [all_extremas[0]]
            
            for i in range(1, len(all_extremas)):
                current_gap = all_extremas[i] - all_extremas[i-1]
                expected_gap = fs / target_freq
                
                # Проверяем, соответствует ли интервал ожидаемой частоте
                gap_ratio = current_gap / expected_gap
                
                if 0.5 <= gap_ratio <= 2.0:  # Более широкий диапазон
                    current_segment.append(all_extremas[i])
                else:
                    # Проверяем завершенный сегмент
                    if len(current_segment) >= min_waves:
                        start_idx = current_segment[0]
                        end_idx = current_segment[-1]
                        
                        # Анализируем сегмент на ритмичность
                        segment_data = filtered_signal[start_idx:end_idx]
                        if self.analyze_segment_rhythmicity(segment_data, fs, target_freq,
                                                          max_freq_variation, max_amp_variation):
                            start_time = start_idx / fs
                            end_time = end_idx / fs
                            duration = end_time - start_time
                            n_waves = len(current_segment) - 1
                            
                            rhythmic_segments.append((start_time, end_time, duration, n_waves))
                    
                    current_segment = [all_extremas[i]]
            
            # Проверяем последний сегмент
            if len(current_segment) >= min_waves:
                start_idx = current_segment[0]
                end_idx = current_segment[-1]
                segment_data = filtered_signal[start_idx:end_idx]
                if self.analyze_segment_rhythmicity(segment_data, fs, target_freq,
                                                  max_freq_variation, max_amp_variation):
                    start_time = start_idx / fs
                    end_time = end_idx / fs
                    duration = end_time - start_time
                    n_waves = len(current_segment) - 1
                    rhythmic_segments.append((start_time, end_time, duration, n_waves))
            
            return rhythmic_segments
            
        except Exception as e:
            st.error(f"Ошибка в поиске ритмических паттернов: {e}")
            return []
    
    def analyze_segment_rhythmicity(self, segment, fs, target_freq, 
                                  max_freq_variation, max_amp_variation):
        """
        Анализирует сегмент на ритмичность с учетом вариабельности
        """
        if len(segment) < 10:
            return False
        
        try:
            # Находим экстремумы в сегменте
            peaks, _ = find_peaks(np.abs(segment), height=0.1, distance=int(fs/target_freq*0.3))
            
            if len(peaks) < 4:  # Минимум 4 экстремума для анализа
                return False
            
            # Анализ интервалов между экстремумами
            intervals = np.diff(peaks) / fs
            frequencies = 1.0 / intervals
            
            # Анализ амплитуд
            amplitudes = np.abs(segment[peaks])
            
            # Проверяем вариабельность (более мягкие критерии)
            if len(frequencies) > 1:
                freq_variation = np.std(frequencies) / np.mean(frequencies)
                amp_variation = np.std(amplitudes) / (np.mean(amplitudes) + 1e-8)
                
                # Мягкие критерии для естественной вариабельности
                freq_ok = freq_variation <= max_freq_variation
                amp_ok = amp_variation <= max_amp_variation
                
                # Дополнительная проверка: средняя частота должна быть в целевом диапазоне
                mean_freq = np.mean(frequencies)
                freq_in_range = (target_freq * 0.6 <= mean_freq <= target_freq * 1.4)
                
                return freq_ok and amp_ok and freq_in_range
            
            return False
            
        except:
            return False
    
    def detect_rhythmic_bursts(self, filtered_signal, fs, low_freq=1, high_freq=40, 
                              threshold_percentile=85, min_duration=0.1, max_duration=2.0,
                              min_waves=6, max_freq_variation=0.4, max_amp_variation=0.7):
        try:
            # Целевая частота для ритмической активности
            target_freq = (low_freq + high_freq) / 2
            
            # Используем улучшенный алгоритм поиска ритмических паттернов
            rhythmic_segments = self.find_rhythmic_patterns(
                filtered_signal, fs, target_freq, min_waves, 
                max_freq_variation, max_amp_variation
            )
            
            # Дополнительная проверка через вейвлет-анализ для подтверждения
            intervals = []
            for start_time, end_time, duration, n_waves in rhythmic_segments:
                if min_duration <= duration <= max_duration:
                    # Проверяем энергию в целевом частотном диапазоне
                    start_idx = int(start_time * fs)
                    end_idx = int(end_time * fs)
                    segment = filtered_signal[start_idx:end_idx]
                    
                    # Простая проверка энергии
                    segment_energy = np.mean(segment**2)
                    if segment_energy > np.percentile(filtered_signal**2, 30):
                        intervals.append((start_time, end_time, duration, n_waves))
            
            # Для визуализации сохраняем некоторые данные
            frequencies = np.linspace(high_freq, low_freq, 50)
            scales = pywt.frequency2scale('cmor1.5-1.0', frequencies) * fs
            coefficients, freqs = pywt.cwt(filtered_signal, scales, 'cmor1.5-1.0', 
                                         sampling_period=1/fs)
            energy = np.abs(coefficients) ** 2
            mean_energy = np.mean(energy, axis=0)
            
            return intervals, mean_energy, coefficients, frequencies
        
        except Exception as e:
            st.error(f"Ошибка в детектировании: {e}")
            return [], np.zeros_like(filtered_signal), np.array([]), np.array([])
    
    def analyze_all_channels(self, low_freq=1, high_freq=40, channels_to_analyze=None,
                           min_waves=6, max_freq_variation=0.4, max_amp_variation=0.7):
        if not hasattr(self, 'multi_channel_data'):
            self.load_edf_data(load_all_channels=True)
        
        if channels_to_analyze is None:
            channels_to_analyze = list(range(len(self.multi_channel_data)))
        
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.all_channel_results = {}
        
        active_channels = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, channel_idx in enumerate(channels_to_analyze):
            channel_data = self.multi_channel_data[channel_idx]
            channel_name = channel_data['name']
            signal = channel_data['data']
            fs = channel_data['fs']
            
            status_text.text(f"Анализ канала: {channel_name} ({idx+1}/{len(channels_to_analyze)})")
            
            # Предобработка сигнала
            filtered_signal = self.preprocess_signal(signal, fs, low_freq, high_freq)
            
            # Детектирование всплесков
            intervals, energy, coefficients, frequencies = self.detect_rhythmic_bursts(
                filtered_signal, fs, low_freq, high_freq,
                min_waves=min_waves,
                max_freq_variation=max_freq_variation,
                max_amp_variation=max_amp_variation
            )
            
            # Сохраняем результаты
            self.all_channel_results[channel_name] = {
                'intervals': intervals,
                'filtered_signal': filtered_signal,
                'energy': energy,
                'coefficients': coefficients,
                'frequencies': frequencies,
                'signal': signal,
                'fs': fs,
                'intervals_count': len(intervals),
                'total_duration': sum(duration for duration, _, _, _ in intervals)
            }
            
            if intervals:
                active_channels.append(channel_name)
            
            progress_bar.progress((idx + 1) / len(channels_to_analyze))
        
        status_text.text("Анализ завершен!")
        return active_channels

def main():
    # Заголовок приложения
    st.title("Поиск ритмичной активности")
    st.markdown("""
    Веб-приложение для обнаружения каналов с ритмичной активностью 
    в записях ЭЭГ формата EDF.
    """)
    
    # Боковая панель для настроек
    st.sidebar.header("Настройки анализа")
    
    # Загрузка файла
    uploaded_file = st.sidebar.file_uploader("Загрузите EDF файл", type=['edf'])
    
    if uploaded_file is not None:
        # Основные параметры анализа
        low_freq = st.sidebar.slider("Нижняя граница частоты (Гц)", 1, 30, 15)
        high_freq = st.sidebar.slider("Верхняя граница частоты (Гц)", 20, 100, 30)
        top_channels = st.sidebar.slider("Количество топ-каналов для отображения", 1, 20, 10)
        threshold = st.sidebar.slider("Порог энергии (%)", 30, 95, 50)
        
        # Расширенные настройки ритмичности
        with st.sidebar.expander("Настройки ритмичности"):
            min_waves = st.slider("Минимальное количество волн", 4, 10, 6)
            max_freq_variation = st.slider("Макс. вариация частоты", 0.2, 1.0, 0.5, 0.05,
                                         help="Допустимое отклонение частоты между волнами")
            max_amp_variation = st.slider("Макс. вариация амплитуды", 0.3, 1.2, 0.8, 0.05,
                                        help="Допустимое отклонение амплитуды между волнами")
        
        # Кнопка запуска анализа
        if st.sidebar.button("Запустить анализ", type="primary"):
            with st.spinner("Анализ ЭЭГ записи..."):
                try:
                    # Инициализация детектора
                    detector = EEGRhythmicActivityDetector(file_buffer=uploaded_file)
                    
                    # Загрузка и анализ данных
                    active_channels = detector.analyze_all_channels(
                        low_freq=low_freq, 
                        high_freq=high_freq,
                        min_waves=min_waves,
                        max_freq_variation=max_freq_variation,
                        max_amp_variation=max_amp_variation
                    )
                    
                    # Основная информация о файле
                    st.header("📊 Основная информация")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Файл", detector.file_info.get('file_name', 'Неизвестно'))
                    with col2:
                        st.metric("Каналов", detector.file_info.get('channels_count', 0))
                    with col3:
                        st.metric("Длительность", f"{detector.file_info.get('duration', 0):.1f} с")
                    with col4:
                        st.metric("Активных каналов", len(active_channels))
                    
                    # Отчет по каналам
                    st.header("📈 Отчет по каналам с активностью")
                    
                    # Собираем данные для отчета
                    report_data = []
                    for channel_name, results in detector.all_channel_results.items():
                        if results['intervals']:
                            total_waves = sum(waves for _, _, _, waves in results['intervals'])
                            
                            report_data.append({
                                'Канал': channel_name,
                                'Интервалы': results['intervals_count'],
                                'Всего волн': total_waves,
                                'Общая длительность (с)': f"{results['total_duration']:.2f}",
                                'Ср. волн/интервал': f"{total_waves/results['intervals_count']:.1f}"
                            })
                    
                    if report_data:
                        # Сортируем по количеству интервалов
                        report_data.sort(key=lambda x: x['Интервалы'], reverse=True)
                        top_report_data = report_data[:top_channels]
                        
                        # Отображаем таблицу
                        st.dataframe(top_report_data, width='stretch')
                        
                        # Визуализация топ-каналов
                        st.header("🎯 Визуализация топ-каналов")
                        
                        # Ограничиваем количество каналов для визуализации
                        channels_to_visualize = [item['Канал'] for item in top_report_data[:6]]
                        
                        # Создаем вкладки для разных типов визуализации
                        tab1, tab2, tab3 = st.tabs(["📊 Сигналы с детекцией", "🌈 Спектрограммы", "📋 Детальный отчет"])
                        
                        with tab1:
                            st.subheader("Сигналы с выделенными интервалами активности")
                            for channel_name in channels_to_visualize:
                                results = detector.all_channel_results[channel_name]
                                fig, ax = plt.subplots(figsize=(12, 3))
                                
                                times = np.arange(len(results['signal'])) / results['fs']
                                ax.plot(times, results['filtered_signal'], 'b-', alpha=0.7, 
                                       label='Фильтрованный ЭЭГ', linewidth=0.8)
                                
                                # Выделение обнаруженных интервалов
                                for start, end, duration, waves in results['intervals']:
                                    ax.axvspan(start, end, alpha=0.3, color='red')
                                    # Подписываем количество волн
                                    ax.text((start + end) / 2, ax.get_ylim()[1] * 0.9, 
                                           f'{waves} волн', ha='center', va='top', 
                                           fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                                                               facecolor="yellow", alpha=0.7))
                                
                                ax.set_title(f'Канал: {channel_name} - {results["intervals_count"]} интервалов')
                                ax.set_xlabel('Время (с)')
                                ax.set_ylabel('Амплитуда (мкВ)')
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                st.pyplot(fig)
                                plt.close()
                        
                        with tab2:
                            st.subheader("Спектрограммы каналов с активностью")
                            
                            colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
                            cmap = LinearSegmentedColormap.from_list('custom_jet', colors, N=256)
                            
                            for channel_name in channels_to_visualize:
                                results = detector.all_channel_results[channel_name]
                                fig, ax = plt.subplots(figsize=(12, 4))
                                
                                times = np.arange(len(results['signal'])) / results['fs']
                                extent = [times[0], times[-1], low_freq, high_freq]
                                
                                im = ax.imshow(np.abs(results['coefficients']), 
                                             extent=extent, aspect='auto', 
                                             origin='lower', cmap=cmap)
                                
                                # Отмечаем обнаруженные интервалы на спектрограмме
                                for start, end, duration, waves in results['intervals']:
                                    ax.axvline(x=start, color='white', linestyle='--', alpha=0.7, linewidth=1)
                                    ax.axvline(x=end, color='white', linestyle='--', alpha=0.7, linewidth=1)
                                    ax.axvspan(start, end, alpha=0.2, color='white')
                                    ax.text((start + end) / 2, high_freq * 0.9, 
                                           f'{waves}', ha='center', va='top', 
                                           color='white', fontweight='bold')
                                
                                ax.set_title(f'Спектрограмма - Канал: {channel_name}')
                                ax.set_xlabel('Время (с)')
                                ax.set_ylabel('Частота (Гц)')
                                plt.colorbar(im, ax=ax, label='Амплитуда')
                                
                                st.pyplot(fig)
                                plt.close()
                        
                        with tab3:
                            st.subheader("Детальный отчет по интервалам")
                            
                            for channel_name in channels_to_visualize:
                                results = detector.all_channel_results[channel_name]
                                
                                with st.expander(f"Канал: {channel_name} ({results['intervals_count']} интервалов)"):
                                    intervals_df = pd.DataFrame([
                                        {
                                            'Начало (с)': f"{start:.2f}",
                                            'Конец (с)': f"{end:.2f}", 
                                            'Длительность (с)': f"{duration:.2f}",
                                            'Кол-во волн': waves,
                                            'Ср. частота (Гц)': f"{(waves-1)/duration:.1f}"
                                        }
                                        for start, end, duration, waves in results['intervals']
                                    ])
                                    st.dataframe(intervals_df, width='stretch')
                    
                    else:
                        st.warning("Ритмическая активность не обнаружена на анализируемых каналах.")
                        st.info("""
                        **Рекомендации:**
                        - Попробуйте ослабить критерии ритмичности в настройках
                        - Увеличьте максимальную вариацию частоты и амплитуды
                        - Уменьшите минимальное количество волн
                        - Проверьте другой частотный диапазон
                        """)
                        
                except Exception as e:
                    st.error(f"Ошибка при анализе: {str(e)}")
    
    else:
        # Инструкция при загрузке
        st.info("""
        ### Инструкция по использованию:
        
        1. **Загрузите EDF файл** с записью ЭЭГ через боковую панель
        2. **Настройте параметры анализа**:
           - Частотный диапазон для детекции (по умолчанию 15-30 Гц)
           - Количество топ-каналов для отображения
           - Порог энергии
        3. **Настройте критерии ритмичности** (в расширенных настройках):
           - Минимальное количество последовательных волн (рекомендуется 4-6)
           - Максимальная вариация частоты (рекомендуется 0.4-0.8)
           - Максимальная вариация амплитуды (рекомендуется 0.6-1.0)
        4. **Нажмите кнопку "Запустить анализ"**
        
        ### Улучшенные критерии ритмической активности:
        - **4-6+ последовательных волн** с естественной вариабельностью
        - **Стабильная частота** (вариация 40-80% допускается)
        - **Стабильная амплитуда** (вариация 60-100% допускается)
        - **Учет как положительных, так и отрицательных пиков**
        - **Адаптивные пороги** для разных типов сигналов
        
        ### Выходные данные:
        - Таблица с статистикой по каналам
        - Визуализация сигналов с выделенными интервалами
        - Спектрограммы для анализа частотно-временных характеристик
        - Детальный отчет по временным интервалам
        """)

if __name__ == "__main__":
    main()
