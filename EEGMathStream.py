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
    page_title="Детекция ритмической активности",
    page_icon="🧠",
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
    
    def is_rhythmic_activity(self, signal_segment, fs, min_peaks=6, max_freq_variation=0.3, max_amp_variation=0.5):
        """
        Проверяет, является ли сегмент сигнала ритмической активностью
        Ритмическая активность: 6+ последовательных волн со стабильной частотой и амплитудой
        """
        try:
            # Находим пики в сегменте
            peaks, properties = find_peaks(signal_segment, height=0, distance=int(fs/(high_freq*2)))
            
            if len(peaks) < min_peaks:
                return False, 0, 0, 0
            
            # Вычисляем интервалы между пиками (периоды)
            peak_intervals = np.diff(peaks) / fs  # в секундах
            peak_frequencies = 1.0 / peak_intervals  # частоты между пиками
            
            # Вычисляем амплитуды пиков
            peak_amplitudes = signal_segment[peaks]
            
            # Проверяем вариабельность частот
            freq_variation = np.std(peak_frequencies) / np.mean(peak_frequencies)
            
            # Проверяем вариабельность амплитуд
            amp_variation = np.std(peak_amplitudes) / np.mean(np.abs(peak_amplitudes))
            
            # Проверяем условия ритмичности
            is_rhythmic = (freq_variation <= max_freq_variation and 
                          amp_variation <= max_amp_variation and
                          len(peaks) >= min_peaks)
            
            return is_rhythmic, len(peaks), freq_variation, amp_variation
            
        except Exception as e:
            return False, 0, 0, 0
    
    def detect_rhythmic_bursts(self, filtered_signal, fs, low_freq=1, high_freq=40, 
                              threshold_percentile=85, min_duration=0.1, max_duration=2.0,
                              min_peaks=6, max_freq_variation=0.3, max_amp_variation=0.5):
        try:
            # Параметры вейвлет-анализа
            frequencies = np.linspace(high_freq, low_freq, 50)
            scales = pywt.frequency2scale('cmor1.5-1.0', frequencies) * fs
            
            # Выполнение непрерывного вейвлет-преобразования
            coefficients, freqs = pywt.cwt(filtered_signal, scales, 'cmor1.5-1.0', 
                                         sampling_period=1/fs)
            
            # Вычисление энергии вейвлет-коэффициентов
            energy = np.abs(coefficients) ** 2
            mean_energy = np.mean(energy, axis=0)
            
            # Пороговое детектирование
            threshold = np.percentile(mean_energy, threshold_percentile)
            above_threshold = mean_energy > threshold
            
            # Поиск интервалов активности
            intervals = []
            start_idx = None
            
            for i, val in enumerate(above_threshold):
                if val and start_idx is None:
                    start_idx = i
                elif not val and start_idx is not None:
                    end_idx = i
                    duration = (end_idx - start_idx) / fs
                    
                    if min_duration <= duration <= max_duration:
                        start_time = start_idx / fs
                        end_time = end_idx / fs
                        
                        # Проверяем, является ли интервал ритмической активностью
                        segment_start = max(0, start_idx - int(0.1 * fs))  # добавляем небольшой запас
                        segment_end = min(len(filtered_signal), end_idx + int(0.1 * fs))
                        signal_segment = filtered_signal[segment_start:segment_end]
                        
                        is_rhythmic, n_peaks, freq_var, amp_var = self.is_rhythmic_activity(
                            signal_segment, fs, min_peaks, max_freq_variation, max_amp_variation
                        )
                        
                        if is_rhythmic:
                            intervals.append((start_time, end_time, duration, n_peaks, freq_var, amp_var))
                    
                    start_idx = None
            
            # Обработка последнего интервала
            if start_idx is not None:
                end_idx = len(above_threshold)
                duration = (end_idx - start_idx) / fs
                if min_duration <= duration <= max_duration:
                    start_time = start_idx / fs
                    end_time = end_idx / fs
                    
                    segment_start = max(0, start_idx - int(0.1 * fs))
                    segment_end = min(len(filtered_signal), end_idx + int(0.1 * fs))
                    signal_segment = filtered_signal[segment_start:segment_end]
                    
                    is_rhythmic, n_peaks, freq_var, amp_var = self.is_rhythmic_activity(
                        signal_segment, fs, min_peaks, max_freq_variation, max_amp_variation
                    )
                    
                    if is_rhythmic:
                        intervals.append((start_time, end_time, duration, n_peaks, freq_var, amp_var))
            
            return intervals, mean_energy, coefficients, frequencies
        
        except Exception as e:
            st.error(f"Ошибка в детектировании: {e}")
            return [], np.zeros_like(filtered_signal), np.array([]), np.array([])
    
    def analyze_all_channels(self, low_freq=1, high_freq=40, channels_to_analyze=None,
                           min_peaks=6, max_freq_variation=0.3, max_amp_variation=0.5):
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
                min_peaks=min_peaks,
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
                'total_duration': sum(duration for duration, _, _, _, _, _ in intervals)
            }
            
            if intervals:
                active_channels.append(channel_name)
            
            progress_bar.progress((idx + 1) / len(channels_to_analyze))
        
        status_text.text("Анализ завершен!")
        return active_channels

def plot_to_html(fig):
    """Конвертирует matplotlib figure в HTML для Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'

def main():
    # Заголовок приложения
    st.title("🧠 Детекция ритмической активности")
    st.markdown("""
    Веб-приложение для автоматического детектирования участков с частой ритмической активностью 
    (острые регулярные волны) в записях ЭЭГ формата EDF.
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
        threshold = st.sidebar.slider("Порог детектирования (%)", 70, 95, 85)
        
        # Расширенные настройки ритмичности
        with st.sidebar.expander("Настройки ритмичности"):
            min_peaks = st.slider("Минимальное количество волн", 3, 12, 6)
            max_freq_variation = st.slider("Макс. вариация частоты", 0.1, 1.0, 0.3, 0.05)
            max_amp_variation = st.slider("Макс. вариация амплитуды", 0.1, 1.0, 0.5, 0.05)
        
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
                        min_peaks=min_peaks,
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
                            total_peaks = sum(peaks for _, _, _, peaks, _, _ in results['intervals'])
                            avg_freq_var = np.mean([freq_var for _, _, _, _, freq_var, _ in results['intervals']])
                            avg_amp_var = np.mean([amp_var for _, _, _, _, _, amp_var in results['intervals']])
                            
                            report_data.append({
                                'Канал': channel_name,
                                'Интервалы': results['intervals_count'],
                                'Волны': total_peaks,
                                'Общая длительность (с)': f"{results['total_duration']:.2f}",
                                'Вариация частоты': f"{avg_freq_var:.3f}",
                                'Вариация амплитуды': f"{avg_amp_var:.3f}"
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
                                ax.plot(times, results['signal'], 'b-', alpha=0.7, label='Исходный ЭЭГ')
                                
                                # Выделение обнаруженных интервалов
                                for start, end, duration, peaks, freq_var, amp_var in results['intervals']:
                                    ax.axvspan(start, end, alpha=0.3, color='red')
                                    # Подписываем количество волн
                                    ax.text((start + end) / 2, ax.get_ylim()[1] * 0.9, 
                                           f'{peaks} волн', ha='center', va='top', fontsize=8)
                                
                                ax.set_title(f'Канал: {channel_name} - {results["intervals_count"]} интервалов, {results["total_duration"]:.2f} с')
                                ax.set_xlabel('Время (с)')
                                ax.set_ylabel('Амплитуда (мкВ)')
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                st.pyplot(fig)
                                plt.close()
                        
                        with tab2:
                            st.subheader("Спектрограммы каналов с активностью")
                            
                            # Создаем кастомную цветовую карту
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
                                for start, end, duration, peaks, freq_var, amp_var in results['intervals']:
                                    ax.axvline(x=start, color='white', linestyle='--', alpha=0.7)
                                    ax.axvline(x=end, color='white', linestyle='--', alpha=0.7)
                                    ax.axvspan(start, end, alpha=0.2, color='white')
                                
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
                                            'Начало (с)': start,
                                            'Конец (с)': end, 
                                            'Длительность (с)': f"{duration:.2f}",
                                            'Кол-во волн': peaks,
                                            'Вариация частоты': f"{freq_var:.3f}",
                                            'Вариация амплитуды': f"{amp_var:.3f}"
                                        }
                                        for start, end, duration, peaks, freq_var, amp_var in results['intervals']
                                    ])
                                    st.dataframe(intervals_df, width='stretch')
                    
                    else:
                        st.info("Ритмическая активность не обнаружена на анализируемых каналах.")
                        
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
           - Порог детектирования
        3. **Настройте критерии ритмичности** (в расширенных настройках):
           - Минимальное количество последовательных волн (не менее 6)
           - Максимальная вариация частоты и амплитуды
        4. **Нажмите кнопку "Запустить анализ"**
        
        ### Критерии ритмической активности:
        - **6+ последовательных волн** с стабильными параметрами
        - **Стабильная частота** (вариация < 30%)
        - **Стабильная амплитуда** (вариация < 50%)
        - **Длительность** от 0.1 до 2.0 секунд
        
        ### Выходные данные:
        - Таблица с статистикой по каналам
        - Визуализация сигналов с выделенными интервалами
        - Спектрограммы для анализа частотно-временных характеристик
        - Детальный отчет по временным интервалам
        """)

if __name__ == "__main__":
    main()
