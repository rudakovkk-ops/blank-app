import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Football Strategy Manager", layout="wide")

st.title("🏆 Менеджер футбольных стратегий: Страта 1")

# Инициализация сессии для хранения истории ставок
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Матч', 'Кэф', 'Результат', 'Прибыль'])

# --- Блок 1: Проверка матча по "Домашнему камбэку" ---
st.sidebar.header("🔍 Анализ матча (Live)")
match_name = st.sidebar.text_input("Название матча", "Ман Сити - Арсенал")
score = st.sidebar.selectbox("Текущий счет", ["0:0", "0:1", "1:0", "Другое"])
minute = st.sidebar.slider("Минута матча", 0, 90, 65)
is_home = st.sidebar.checkbox("Фаворит играет ДОМА", value=True)
motivation = st.sidebar.checkbox("Высокая турнирная мотивация", value=True)
pressure = st.sidebar.slider("Индекс давления (опасные атаки за 10 мин)", 0, 10, 4)

# Кнопка проверки
if st.sidebar.button("Проверить по чек-листу"):
    passed = (score == "0:1" and 60 <= minute <= 75 and is_home and motivation and pressure >= 3)
    
    if passed:
        st.success(f"✅ МАТЧ ПОДХОДИТ! Рекомендуемая ставка на ничью (X) в {match_name}")
        st.balloons()
    else:
        st.error("❌ НЕ ПОДХОДИТ. Одно или несколько условий чек-листа не выполнены.")

# --- Блок 2: Калькулятор и Журнал ставок ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Добавить результат")
    with st.form("bet_form"):
        res_match = st.text_input("Матч", match_name)
        res_odds = st.number_input("Коэффициент", min_value=1.0, value=4.2)
        res_status = st.selectbox("Результат", ["Выигрыш", "Проигрыш"])
        submit = st.form_submit_button("Записать в базу")
        
        if submit:
            profit = (res_odds - 1) if res_status == "Выигрыш" else -1
            new_row = {'Матч': res_match, 'Кэф': res_odds, 'Результат': res_status, 'Прибыль': profit}
            st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([new_row])], ignore_index=True)

with col2:
    st.subheader("📈 Динамика профита")
    if not st.session_state.history.empty:
        # Расчет кумулятивной прибыли
        st.session_state.history['Кумулятив'] = st.session_state.history['Прибыль'].cumsum()
        
        # График
        fig, ax = plt.subplots()
        ax.plot(st.session_state.history['Кумулятив'], marker='o', color='#2ca02c')
        ax.axhline(0, color='black', linestyle='--')
        ax.set_ylabel("Чистая прибыль (в ставках)")
        ax.set_xlabel("Номер ставки")
        st.pyplot(fig)
    else:
        st.info("История ставок пока пуста. Добавьте первую ставку слева.")

# --- Блок 3: База знаний (Notion-style) ---
with st.expander("📖 Справочник: Страта 1 'Домашний камбэк'"):
    st.markdown("""
    **Основные параметры:**
    * **Вход:** Счет 0:1, 60-75 минута.
    * **Фильтр 1:** Только домашний фаворит.
    * **Фильтр 2:** Мотивация (битва за титул/ЛЧ).
    * **Фильтр 3:** Наличие давления (минимум 3 опасных момента за 10 мин).
    * **Целевой ROI:** ~76%.
    """)

# Отображение таблицы
st.subheader("📋 Журнал последних ставок")
st.dataframe(st.session_state.history, use_container_width=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- Вспомогательные функции аналитики ---
def calculate_metrics(df):
    if df.empty:
        return 0, 0, 0
    total_bets = len(df)
    total_profit = df['Прибыль'].sum()
    roi = (total_profit / total_bets) * 100
    win_rate = (len(df[df['Результат'] == 'Выигрыш']) / total_bets) * 100
    return total_profit, roi, win_rate

def convert_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Stats')
    return output.getvalue()

# --- Секция Дашборда (добавить после таблицы истории) ---
st.divider()
st.subheader("📊 Аналитика и Отчетность")

if not st.session_state.history.empty:
    profit, roi, wr = calculate_metrics(st.session_state.history)
    
    # Виджеты с ключевыми показателями
    m1, m2, m3 = st.columns(3)
    m1.metric("Общий профит", f"{profit:.2f} ед.", delta=f"{profit:.2f}")
    m2.metric("Текущий ROI", f"{roi:.1f}%")
    m3.metric("Win Rate", f"{wr:.1f}%")

    # Кнопка выгрузки
    excel_data = convert_to_excel(st.session_state.history)
    st.download_button(
        label="📥 Скачать полный отчет в Excel",
        data=excel_data,
        file_name=f"strategy_report_{pd.Timestamp.now().strftime('%Y-%m-%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.warning("Недостаточно данных для расчета аналитики.")
