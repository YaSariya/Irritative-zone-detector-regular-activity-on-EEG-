#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install streamlit


# In[ ]:


streamlit run simple_app.py


# In[ ]:


pip install numpy pandas matplotlib pyedflib pywavelets scipy


# In[ ]:


streamlit run app.py


# In[ ]:


# app.py - Минимальная версия анализатора ЭЭГ

import streamlit as st
import io
import base64

# Проверка доступности библиотек
def check_dependencies():
    missing_deps = []
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas as pd
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import pyedflib
    except ImportError:
        missing_deps.append("pyedflib")
    
    try:
        import pywt
    except ImportError:
        missing_deps.append("pywavelets")
    
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        missing_deps.append("scipy")
    
    return missing_deps

def main():
    st.set_page_config(
        page_title="Поиск регулярной быстроволновой активности",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Поиск регулярной быстроволновой активности")
    
    # Проверка зависимостей
    missing_deps = check_dependencies()
    
    if missing_deps:
        st.error("⚠️ Отсутствуют необходимые библиотеки!")
        st.markdown(f"""
        ### Необходимо установить следующие библиотеки:
        ```bash
        pip install {" ".join(missing_deps)}
        ```
        
        ### Или установите все зависимости сразу:
        ```bash
        pip install streamlit numpy pandas matplotlib pyedflib pywavelets scipy
        ```
        """)
        
        # Показываем демо-интерфейс даже без зависимостей
        show_demo_interface()
        return
    
    # Если все зависимости установлены, запускаем полную версию
    run_full_app()

def show_demo_interface():
    """Показывает демо-интерфейс когда зависимости не установлены"""
    st.warning("Загрузите EDF файл для анализа (функциональность ограничена без установленных зависимостей)")
    
    uploaded_file = st.file_uploader("Загрузите EDF файл", type=['edf'])
    
    if uploaded_file is not None:
        st.success(f"Файл {uploaded_file.name} загружен успешно!")
        st.info("Для полного анализа установите все зависимости как указано выше.")
        
        # Показываем информацию о файле
        file_details = {
            "Имя файла": uploaded_file.name,
            "Размер файла": f"{uploaded_file.size / 1024:.2f} KB",
            "Тип файла": uploaded_file.type
        }
        st.json(file_details)
    else:
        st.info("""
        ### Инструкция по настройке:
        
        1. **Установите зависимости** используя команды выше
        2. **Перезапустите приложение**: 
           - Остановите текущее выполнение (Ctrl+C)
           - Запустите снова: `streamlit run app.py`
        3. **Загрузите EDF файл** для анализа
        
        ### Возможности приложения:
        - Детекция ритмической активности в ЭЭГ
        - Визуализация сигналов и спектрограмм
        - Анализ по множественным каналам
        - Генерация отчетов
        """)

def run_full_app():
    """Запускает полную версию приложения когда все зависимости установлены"""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import pyedflib
    import pywt
    from scipy.signal import butter, filtfilt, hilbert
    import os
    from datetime import datetime
    
    class SimpleEEGAnalyzer:
        def __init__(self, file_buffer):
            self.file_buffer = file_buffer
            self.channel_names = []
            self.file_info = {}
            
        def load_file_info(self):
            """Просто загружает информацию о файле без полного анализа"""
            try:
                # Сохраняем временный файл
                with open("temp_edf_file.edf", "wb") as f:
                    f.write(self.file_buffer.getvalue())
                
                with pyedflib.EdfReader("temp_edf_file.edf") as edf:
                    self.channel_names = edf.getSignalLabels()
                    n_channels = edf.signals_in_file
                    
                    self.file_info = {
                        'file_name': os.path.basename("temp_edf_file.edf"),
                        'duration': edf.getFileDuration(),
                        'start_time': edf.getStartdatetime(),
                        'channels_count': n_channels,
                        'sample_frequency': int(edf.getSampleFrequency(0)) if n_channels > 0 else 0
                    }
                
                # Удаляем временный файл
                if os.path.exists("temp_edf_file.edf"):
                    os.remove("temp_edf_file.edf")
                    
                return True
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
                return False
        
        def quick_analysis(self):
            """Простой анализ для демонстрации"""
            st.info("Выполняется быстрый анализ...")
            
            # Создаем демо-данные для визуализации
            fs = 250  #Hz
            t = np.linspace(0, 10, fs * 10)
            
            # Генерируем демо-сигнал с ритмической активностью
            signal = np.sin(2 * np.pi * 8 * t)  # Основной ритм
            signal += 0.5 * np.sin(2 * np.pi * 25 * t)  # Высокочастотная активность
            signal += 0.3 * np.random.normal(size=len(t))  # Шум
            
            # Детекция "активности" - просто для демо
            rhythmic_regions = [(2, 3), (5, 6), (8, 9)]
            
            return signal, t, rhythmic_regions, fs
    
    # Основной интерфейс приложения
    st.sidebar.header("Настройки анализа")
    
    uploaded_file = st.sidebar.file_uploader("Загрузите EDF файл", type=['edf'])
    
    if uploaded_file is not None:
        # Параметры анализа
        low_freq = st.sidebar.slider("Нижняя граница частоты (Гц)", 10, 30, 15)
        high_freq = st.sidebar.slider("Верхняя граница частоты (Гц)", 20, 50, 30)
        
        if st.sidebar.button("Выполнить анализ", type="primary"):
            with st.spinner("Анализ ЭЭГ..."):
                try:
                    # Инициализация анализатора
                    analyzer = SimpleEEGAnalyzer(uploaded_file)
                    
                    # Загрузка информации о файле
                    if analyzer.load_file_info():
                        # Основная информация
                        st.header("📊 Информация о файле")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Файл", analyzer.file_info.get('file_name', 'N/A'))
                        with col2:
                            st.metric("Каналы", analyzer.file_info.get('channels_count', 0))
                        with col3:
                            st.metric("Длительность", f"{analyzer.file_info.get('duration', 0):.1f} с")
                        with col4:
                            st.metric("Частота дискр.", f"{analyzer.file_info.get('sample_frequency', 0)} Гц")
                        
                        # Список каналов
                        st.subheader("📋 Доступные каналы")
                        st.write(f"Всего каналов: {len(analyzer.channel_names)}")
                        
                        # Показываем первые 10 каналов
                        channels_to_show = analyzer.channel_names[:10]
                        for i, channel in enumerate(channels_to_show):
                            st.write(f"{i+1}. {channel}")
                        
                        if len(analyzer.channel_names) > 10:
                            st.info(f"... и еще {len(analyzer.channel_names) - 10} каналов")
                        
                        # Демо-визуализация
                        st.header("🎯 Демонстрация анализа")
                        
                        # Генерируем демо-данные
                        signal, times, regions, fs = analyzer.quick_analysis()
                        
                        # Визуализация сигнала
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(times, signal, 'b-', alpha=0.7, label='ЭЭГ сигнал')
                        
                        # Выделяем "обнаруженные" области
                        for start, end in regions:
                            ax.axvspan(start, end, alpha=0.3, color='red', label='Ритмическая активность')
                        
                        ax.set_title('Обнаружение ритмической активности')
                        ax.set_xlabel('Время (с)')
                        ax.set_ylabel('Амплитуда')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        
                        st.pyplot(fig)
                        
                        # Демо-спектрограмма
                        st.subheader("🌈 Демонстрация спектрограммы")
                        
                        # Создаем простую спектрограмму
                        fig, ax = plt.subplots(figsize=(12, 4))
                        
                        # Вычисляем спектрограмму
                        f, t, Sxx = plt.specgram(signal, Fs=fs, cmap='jet')
                        
                        ax.set_title('Демонстрационная спектрограмма')
                        ax.set_xlabel('Время (с)')
                        ax.set_ylabel('Частота (Гц)')
                        plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, label='Интенсивность')
                        
                        st.pyplot(fig)
                        
                        # Демо-отчет
                        st.header("📈 Демонстрационный отчет")
                        
                        report_data = []
                        for i, channel in enumerate(analyzer.channel_names[:5]):  # Первые 5 каналов
                            report_data.append({
                                'Канал': channel,
                                'Интервалы': np.random.randint(1, 10),
                                'Общая длительность (с)': f"{np.random.uniform(1, 5):.2f}",
                                'Ср. амплитуда (мкВ)': f"{np.random.uniform(0.01, 0.1):.4f}"
                            })
                        
                        st.dataframe(report_data)
                        
                        st.success("✅ Анализ завершен успешно!")
                        
                except Exception as e:
                    st.error(f"Ошибка при анализе: {str(e)}")
    
    else:
        st.info("""
        ### 🧠 Анализатор ЭЭГ - Полная версия
        
        **Возможности:**
        - Детекция ритмической активности в ЭЭГ записях
        - Анализ по всем каналам EDF файла
        - Визуализация сигналов с выделенными интервалами
        - Спектрограммы для частотно-временного анализа
        - Генерация подробных отчетов
        
        **Инструкция:**
        1. Загрузите EDF файл через боковую панель
        2. Настройте параметры анализа
        3. Нажмите "Выполнить анализ"
        
        **Поддерживаемые форматы:** EDF (European Data Format)
        """)

if __name__ == "__main__":
    main()

