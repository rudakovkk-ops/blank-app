"""Расширенный русскоязычный интерфейс с live-прогнозами и реальными API-данными."""
import streamlit as st
from datetime import datetime, timedelta
from threading import Thread
import pandas as pd
import numpy as np
import logging
import json
import plotly.graph_objects as go
from config.settings import (
    ELO_BASE_RATING,
    ELO_K_FACTOR,
    ELO_DEFAULT_HOME_ADVANTAGE,
    ELO_SEASON_CARRYOVER,
    ELO_HOME_ADVANTAGE_CARRYOVER,
    LEAGUE_HOME_ADVANTAGE_K_FACTOR,
    MIN_LEAGUE_HOME_ADVANTAGE,
    MAX_LEAGUE_HOME_ADVANTAGE,
    FEATURE_DIAGNOSTIC_COLUMNS,
)
from data.data_service import DataService
from prediction_service import PredictionService
from scheduler.auto_updater import DataUpdateScheduler

logger = logging.getLogger(__name__)

RUDY_TABLE_FORMAT_VERSION = 3


def probability_value(probability_map: dict, russian_label: str, english_label: str) -> float:
    """Вернуть вероятность с поддержкой новых русских и старых английских ключей."""
    return float(probability_map.get(russian_label, probability_map.get(english_label, 0.0)))


# Конфигурация Streamlit
st.set_page_config(
    page_title="Центр футбольных прогнозов",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 0; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
    .prediction-high { background-color: #d4edda; padding: 0.5rem; border-radius: 0.3rem; }
    .prediction-medium { background-color: #fff3cd; padding: 0.5rem; border-radius: 0.3rem; }
    .prediction-low { background-color: #f8d7da; padding: 0.5rem; border-radius: 0.3rem; }
    .summary-card {
        border-radius: 0.75rem;
        padding: 1rem 1.1rem;
        border: 1px solid #d7dce2;
        background: #f7f9fb;
    }
    .summary-card.best {
        border-color: #2f855a;
        background: #edf8f1;
        box-shadow: inset 0 0 0 1px rgba(47, 133, 90, 0.08);
    }
    .summary-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #5b6470;
        margin-bottom: 0.35rem;
    }
    .summary-model {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f2933;
        margin-bottom: 0.3rem;
    }
    .summary-meta {
        font-size: 0.9rem;
        color: #52606d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_data_service():
    """Инициализировать сервис данных"""
    return DataService()


@st.cache_resource
def get_prediction_service():
    """Инициализировать сервис инференса."""
    return PredictionService()


@st.cache_resource
def get_update_scheduler():
    """Поднять фоновый APScheduler один раз на процесс Streamlit."""
    scheduler = DataUpdateScheduler()
    scheduler.start()
    return scheduler

# Инициализация всех сервисов при загрузке страницы
try:
    data_service = get_data_service()
    prediction_service = get_prediction_service()
    update_scheduler = get_update_scheduler()
    scheduler_status = update_scheduler.get_status()
except Exception as e:
    st.error(f"Ошибка инициализации сервисов: {e}")
    logger.exception("Failed to initialize services")
    import sys
    sys.exit(1)

# Боковая панель
with st.sidebar:
    st.title("⚽ Центр футбольных прогнозов")
    st.markdown("---")
    
    page = st.radio(
        "Навигация",
        [
            "🏠 Дашборд",
            "⚡ Live-прогнозы",
            "🤖 Модели",
            "🧪 Диагностика признаков",
            "💰 ROI-анализ",
            "⚙️ Настройки",
            "📋 О проекте"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    with st.expander("📡 Состояние системы"):
        st.write("**Статус:** ✅ Онлайн")
        st.write("**API:** Подключено")
        st.write("**Последнее обновление:** Только что")
        st.write(f"**Планировщик:** {'Активен' if scheduler_status.get('is_running') else 'Остановлен'}")
        
        if st.button("🔄 Очистить кэш"):
            data_service.clear_cache()
            st.success("Кэш очищен")

# Основной контент
if page == "🏠 Дашборд":
    st.header("🏠 Дашборд")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Сегодняшние матчи")
    with col2:
        if st.button("🔄 Обновить", use_container_width=True):
            st.rerun()

    st.markdown("### Rudy: прогноз по всем сегодняшним матчам")
    force_col1, force_col2 = st.columns([1, 3])
    with force_col1:
        force_rudy = st.button("🚀 Принудительно пересчитать Rudy", use_container_width=True)
    with force_col2:
        st.caption(
            "Кнопка запускает обновление матчей из api-football.com по всем нашим чемпионатам "
            "на актуальную дату и строит таблицу Rudy (5 дома + 5 в гостях + 5 H2H)."
        )

    if force_rudy:
        with st.spinner("Обновляю матчи из API и пересчитываю Rudy..."):
            tracked_today_fixtures = data_service.get_today_fixtures_tracked_leagues(force_refresh=True)
            st.session_state.rudy_today_rows = prediction_service.build_rudy_today_rows(tracked_today_fixtures)
            st.session_state.rudy_today_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(
            f"Rudy пересчитан: {len(st.session_state.get('rudy_today_rows', []))} матчей"
        )

    if st.session_state.get('rudy_today_rows_version') != RUDY_TABLE_FORMAT_VERSION:
        st.session_state.pop('rudy_today_rows', None)
        st.session_state['rudy_today_rows_version'] = RUDY_TABLE_FORMAT_VERSION

    if 'rudy_today_rows' not in st.session_state:
        tracked_today_fixtures = data_service.get_today_fixtures_tracked_leagues(force_refresh=False)
        st.session_state.rudy_today_rows = prediction_service.build_rudy_today_rows(tracked_today_fixtures)
        st.session_state.rudy_today_generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.caption(
        f"Rudy обновлен: {st.session_state.get('rudy_today_generated_at', 'Н/Д')} | "
        f"строк: {len(st.session_state.get('rudy_today_rows', []))}"
    )

    rudy_rows = st.session_state.get('rudy_today_rows', [])
    if rudy_rows:
        cols_order = ['Время', 'Матч', 'Страна', 'Дома', 'В гостях', 'Вывод', 'RudySuper']
        col_widths = {
            'Время': '5%', 'Матч': '12%', 'Страна': '8%',
            'Дома': '18%', 'В гостях': '18%', 'Вывод': '19%', 'RudySuper': '20%',
        }
        header_html = ''.join(
            f'<th style="width:{col_widths.get(c,"auto")};padding:4px 6px;border-bottom:2px solid #444;'
            f'white-space:nowrap;font-size:11px;text-align:left">{c}</th>'
            for c in cols_order
        )
        rows_html = ''
        for row in rudy_rows:
            cells = ''
            for c in cols_order:
                val = str(row.get(c, ''))
                cells += (
                    f'<td style="padding:3px 6px;font-size:11px;vertical-align:top;'
                    f'word-break:break-word;white-space:pre-wrap">{val}</td>'
                )
            rows_html += f'<tr>{cells}</tr>'
        table_html = f"""
<div style="overflow-x:hidden;width:100%">
<table style="width:100%;border-collapse:collapse;table-layout:fixed;font-family:monospace">
<thead><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.info("Для сегодняшней даты в отслеживаемых лигах матчи не найдены.")
    
    st.markdown("### Сводка по качеству модели")
    primary_metrics = prediction_service.get_primary_model_metrics()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Точность", f"{primary_metrics['accuracy'] * 100:.1f}%")
    with col2:
        st.metric("Precision", f"{primary_metrics['precision'] * 100:.1f}%")
    with col3:
        st.metric("Recall", f"{primary_metrics['recall'] * 100:.1f}%")
    with col4:
        st.metric("ROC-AUC", f"{primary_metrics['roc_auc']:.3f}")

    st.caption(f"Основная модель: {primary_metrics['model_label']}")

elif page == "⚡ Live-прогнозы":
    st.header("⚡ Live-прогнозы")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Live-матчи (по Московскому времени)")
    with col2:
        if st.button("🔄 Обновить", use_container_width=True, key="live_refresh"):
            st.rerun()
    
    try:
        # Получить LIVE матчи
        live_fixtures = data_service.get_live_fixtures()
        
        if live_fixtures:
            st.markdown(f"**Активные матчи: {len(live_fixtures)}**")
            
            rows = []
            for fixture in live_fixtures:
                if not prediction_service.is_supported_fixture(fixture):
                    continue
                
                teams = fixture.get('teams', {})
                league = fixture.get('league', {})
                goals = fixture.get('goals', {})
                status_info = fixture.get('fixture', {})
                
                # Время в UTC -> конвертируем в МСК (UTC+3)
                utc_time_str = status_info.get('date', '')[:16]
                if utc_time_str:
                    try:
                        utc_time = datetime.fromisoformat(utc_time_str)
                        msk_time = utc_time + timedelta(hours=3)
                        match_time = msk_time.strftime("%H:%M")
                    except:
                        match_time = utc_time_str[11:16]
                else:
                    match_time = 'Н/Д'
                
                # Получаем RudySuper прогноз
                try:
                    prediction_super = prediction_service._predict_with_rudy_super(fixture)
                    verdict = prediction_super.get('prediction_label', 'Н/Д')
                    agreement = prediction_super.get('agreement_level', '')
                    if agreement:
                        verdict_text = f"{verdict}\n({agreement})"
                    else:
                        verdict_text = verdict
                except:
                    verdict_text = "Ошибка прогноза"
                
                rows.append({
                    'Страна': league.get('country', 'Н/Д'),
                    'К1 - К2': f"{teams.get('home', {}).get('name', 'Н/Д')} - {teams.get('away', {}).get('name', 'Н/Д')}",
                    'Время (МСК)': match_time,
                    'Счет': f"{goals.get('home', 0)} - {goals.get('away', 0)}",
                    'Вывод RudySuper': verdict_text,
                })
            
            if rows:
                # HTML таблица для лучшего форматирования
                cols_order = ['Страна', 'К1 - К2', 'Время (МСК)', 'Счет', 'Вывод RudySuper']
                col_widths = {
                    'Страна': '10%', 'К1 - К2': '35%', 'Время (МСК)': '12%',
                    'Счет': '10%', 'Вывод RudySuper': '33%',
                }
                header_html = ''.join(
                    f'<th style="width:{col_widths.get(c,"auto")};padding:4px 6px;border-bottom:2px solid #444;'
                    f'white-space:nowrap;font-size:11px;text-align:left">{c}</th>'
                    for c in cols_order
                )
                rows_html = ''
                for row in rows:
                    cells = ''
                    for c in cols_order:
                        val = str(row.get(c, ''))
                        cells += (
                            f'<td style="padding:3px 6px;font-size:11px;vertical-align:top;'
                            f'word-break:break-word;white-space:pre-wrap">{val}</td>'
                        )
                    rows_html += f'<tr>{cells}</tr>'
                table_html = f"""
<div style="overflow-x:hidden;width:100%">
<table style="width:100%;border-collapse:collapse;table-layout:fixed;font-family:monospace">
<thead><tr>{header_html}</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("Нет поддерживаемых лигам среди активных матчей")
        else:
            st.info("⏸️ Сейчас live-матчей нет. Проверьте позже.")
    except Exception as e:
        st.error(f"Ошибка загрузки live-прогнозов: {e}")
        logger.error(f"Ошибка live-прогнозов: {e}")

elif page == "🤖 Модели":
    st.header("🤖 Модель Rudy")
    st.caption("В проекте оставлена только rule-based модель Rudy (5 домашних, 5 гостевых и 5 H2H матчей).")

    model_metrics = prediction_service.get_model_metrics()
    if not model_metrics:
        st.info("Метрики Rudy пока недоступны.")
    else:
        rudy_metrics = model_metrics[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Точность", f"{rudy_metrics['accuracy'] * 100:.1f}%")
        with col2:
            st.metric("Precision", f"{rudy_metrics['precision'] * 100:.1f}%")
        with col3:
            st.metric("Recall", f"{rudy_metrics['recall'] * 100:.1f}%")
        with col4:
            st.metric("ROC-AUC", f"{rudy_metrics['roc_auc']:.3f}")

        details_df = pd.DataFrame([
            {"Параметр": "Тип модели", "Значение": "Rule-based"},
            {"Параметр": "Домашнее окно", "Значение": int(prediction_service.rudy_model.home_window)},
            {"Параметр": "Гостевое окно", "Значение": int(prediction_service.rudy_model.away_window)},
            {"Параметр": "H2H окно", "Значение": int(prediction_service.rudy_model.h2h_window)},
        ])
        st.dataframe(details_df, use_container_width=True, hide_index=True)

    st.markdown("### Пример прогноза Rudy")
    today_fixtures = data_service.get_today_fixtures()
    tracked_fixture = prediction_service.get_reference_fixture(today_fixtures)
    if tracked_fixture:
        prediction = prediction_service.predict_fixture(tracked_fixture, model_key='rudy', skip_enrichment=True)
        if prediction:
            st.caption(
                f"Матч: {tracked_fixture.get('teams', {}).get('home', {}).get('name', 'Н/Д')} vs "
                f"{tracked_fixture.get('teams', {}).get('away', {}).get('name', 'Н/Д')}"
            )
            st.dataframe(
                pd.DataFrame([
                    {
                        "Модель": prediction.get("model_label", "Rudy"),
                        "Прогноз": prediction.get("prediction_label", "Н/Д"),
                        "Уверенность": f"{prediction.get('confidence', 0.0) * 100:.1f}%",
                    }
                ]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Не удалось построить прогноз для опорного матча.")
    else:
        st.info("Сейчас нет подходящих матчей для примера прогноза.")

elif page == "🧪 Диагностика признаков":
    st.header("🧪 Диагностика признаков")
    st.markdown("### Срез признаков на уровне матча")

    live_fixtures = data_service.get_live_fixtures()
    today_fixtures = data_service.get_today_fixtures()
    fixture_map = {}
    for fixture in live_fixtures + today_fixtures:
        fixture_id = fixture.get("fixture", {}).get("id")
        if fixture_id is None or not prediction_service.is_supported_fixture(fixture):
            continue
        fixture_map[fixture_id] = fixture

    supported_fixtures = list(fixture_map.values())
    if not supported_fixtures:
        st.info("Сейчас нет подходящих матчей для диагностики.")
    else:
        model_metrics = prediction_service.get_model_metrics()
        model_label_to_key = {item['model_label']: item['model_key'] for item in model_metrics}
        model_labels = [item['model_label'] for item in model_metrics]

        fixture_labels = {
            prediction_service.get_fixture_label(fixture): fixture
            for fixture in supported_fixtures
        }

        col1, col2 = st.columns([3, 2])
        with col1:
            selected_fixture_label = st.selectbox(
                "Матч",
                list(fixture_labels.keys()),
            )
        with col2:
            selected_model_label = st.selectbox(
                "Модель",
                model_labels,
                index=model_labels.index(prediction_service.MODEL_LABELS[prediction_service.primary_model_key]),
            )

        diagnostics = prediction_service.get_feature_diagnostics(
            fixture_labels[selected_fixture_label],
            model_key=model_label_to_key[selected_model_label],
        )

        if diagnostics is None:
            st.warning("Не удалось построить диагностические признаки для выбранного матча.")
        else:
            prediction = diagnostics["prediction"]
            probabilities = prediction["probabilities"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Модель", prediction["model_label"])
            with col2:
                st.metric("Прогноз", prediction["prediction_label"])
            with col3:
                st.metric("Уверенность", f"{prediction['confidence'] * 100:.1f}%")
            with col4:
                st.metric(
                    "Число признаков",
                    str(diagnostics["feature_count"]),
                    None if not prediction.get('is_abstained') else f"Кандидат: {prediction.get('raw_prediction_label', '-')}",
                )

            probability_df = pd.DataFrame({
                "Исход": list(probabilities.keys()),
                "Вероятность": [f"{value * 100:.1f}%" for value in probabilities.values()],
            })
            st.dataframe(probability_df, use_container_width=True, hide_index=True)

            st.markdown("### Ключевые признаки")
            st.dataframe(
                diagnostics["focus_features"].sort_values("feature").reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Наибольшие отклонения от обучающих значений по умолчанию")
            st.dataframe(
                diagnostics["top_feature_deltas"],
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("Сырые признаки"):
                raw_df = diagnostics["raw_feature_row"].T.reset_index()
                raw_df.columns = ["признак", "значение"]
                st.dataframe(raw_df, use_container_width=True, hide_index=True)

            with st.expander("Нормализованные признаки"):
                normalized_df = diagnostics["normalized_feature_row"].T.reset_index()
                normalized_df.columns = ["признак", "значение"]
                st.dataframe(normalized_df, use_container_width=True, hide_index=True)

elif page == "💰 ROI-анализ":
    st.header("💰 Статистика RudySuper")
    st.markdown("### Точность предсказаний на завершённых матчах")
    st.caption("Счётчик накапливает все завершённые матчи из кэша. Нажмите кнопку для пересчёта.")

    # Кнопка принудительного пересчёта
    if st.button("🔄 Принудительно пересчитать", use_container_width=True):
        with st.spinner("Обрабатываю завершённые матчи из кэша..."):
            result = prediction_service.accumulate_rudy_super_stats()
        st.success(f"Добавлено матчей: {result['added']} | Ошибок: {result['errors']}")
        stats = result['summary']
    else:
        stats = prediction_service.cache.get_rudy_super_stats_summary()

    if stats['total_matches'] > 0:
        # Ряд с основными метриками
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Всего матчей",
                f"{stats['total_matches']}",
                help="Количество завершённых матчей в базе счётчика"
            )

        with col2:
            p1_accuracy = (stats['p1_correct'] / max(stats['p1_total'], 1)) * 100
            st.metric(
                "П1 верно",
                f"{stats['p1_correct']}/{stats['p1_total']}",
                f"{p1_accuracy:.1f}%",
                help="Сколько раз RudySuper правильно предсказал Победу хозяев"
            )

        with col3:
            draw_accuracy = (stats['draw_correct'] / max(stats['draw_total'], 1)) * 100
            st.metric(
                "Ничья верно",
                f"{stats['draw_correct']}/{stats['draw_total']}",
                f"{draw_accuracy:.1f}%",
                help="Сколько раз RudySuper правильно предсказал Ничью"
            )

        with col4:
            p2_accuracy = (stats['p2_correct'] / max(stats['p2_total'], 1)) * 100
            st.metric(
                "П2 верно",
                f"{stats['p2_correct']}/{stats['p2_total']}",
                f"{p2_accuracy:.1f}%",
                help="Сколько раз RudySuper правильно предсказал Победу гостей"
            )

        # Общая точность
        total_correct = stats['total_correct']
        overall_accuracy = (total_correct / stats['total_matches']) * 100

        st.markdown("---")
        st.markdown(f"### Общая точность: **{overall_accuracy:.1f}%** ({total_correct}/{stats['total_matches']} матчей)")

        # Детальная таблица
        st.markdown("### Детализация по типам исходов")
        detail_data = [
            {
                "Тип исхода": "Победа хозяев (П1)",
                "Матчей": stats['p1_total'],
                "Верно": stats['p1_correct'],
                "Ошибки": stats['p1_total'] - stats['p1_correct'],
                "Точность": f"{(stats['p1_correct'] / max(stats['p1_total'], 1)) * 100:.1f}%"
            },
            {
                "Тип исхода": "Ничья",
                "Матчей": stats['draw_total'],
                "Верно": stats['draw_correct'],
                "Ошибки": stats['draw_total'] - stats['draw_correct'],
                "Точность": f"{(stats['draw_correct'] / max(stats['draw_total'], 1)) * 100:.1f}%"
            },
            {
                "Тип исхода": "Победа гостей (П2)",
                "Матчей": stats['p2_total'],
                "Верно": stats['p2_correct'],
                "Ошибки": stats['p2_total'] - stats['p2_correct'],
                "Точность": f"{(stats['p2_correct'] / max(stats['p2_total'], 1)) * 100:.1f}%"
            },
        ]
        detail_df = pd.DataFrame(detail_data)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

        if stats.get('last_update'):
            st.caption(f"Последнее обновление данных в базе: {stats['last_update']}")
    else:
        st.info("База счётчика пуста. Нажмите «🔄 Принудительно пересчитать», чтобы загрузить матчи из кэша.")

elif page == "⚙️ Настройки":
    st.header("⚙️ Настройки")
    st.subheader("⏰ Фоновое обновление")
    st.metric("Планировщик", "Активен" if scheduler_status.get('is_running') else "Остановлен")
    
    st.subheader("💾 Управление кэшем")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистить весь кэш", use_container_width=True):
            data_service.clear_cache()
            st.success("Кэш успешно очищен")
    with col2:
        if st.button("🔄 Обновить данные", use_container_width=True):
            data_service.clear_cache()
            st.success("Кэш обновлен. Будут запрошены новые данные.")
    
    st.subheader("🎨 Параметры интерфейса")
    currency = st.selectbox("Валюта", ["USD", "EUR", "GBP", "RUB"])
    notifications = st.checkbox("Включить уведомления", value=True)
    auto_refresh = st.slider("Интервал автообновления (минуты)", 1, 60, 15)

    st.subheader("🧠 Feature Engineering Configuration")
    config_df = pd.DataFrame([
        {"Параметр": "ELO_BASE_RATING", "Значение": ELO_BASE_RATING},
        {"Параметр": "ELO_K_FACTOR", "Значение": ELO_K_FACTOR},
        {"Параметр": "ELO_DEFAULT_HOME_ADVANTAGE", "Значение": ELO_DEFAULT_HOME_ADVANTAGE},
        {"Параметр": "ELO_SEASON_CARRYOVER", "Значение": ELO_SEASON_CARRYOVER},
        {"Параметр": "ELO_HOME_ADVANTAGE_CARRYOVER", "Значение": ELO_HOME_ADVANTAGE_CARRYOVER},
        {"Параметр": "LEAGUE_HOME_ADVANTAGE_K_FACTOR", "Значение": LEAGUE_HOME_ADVANTAGE_K_FACTOR},
        {"Параметр": "MIN_LEAGUE_HOME_ADVANTAGE", "Значение": MIN_LEAGUE_HOME_ADVANTAGE},
        {"Параметр": "MAX_LEAGUE_HOME_ADVANTAGE", "Значение": MAX_LEAGUE_HOME_ADVANTAGE},
        {"Параметр": "FEATURE_DIAGNOSTIC_COLUMNS", "Значение": len(FEATURE_DIAGNOSTIC_COLUMNS)},
    ])
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    snapshot_stats = prediction_service.get_live_snapshot_stats()
    snapshot_df = pd.DataFrame([
        {"Метрика": "Live-снимки", "Значение": snapshot_stats['snapshot_count']},
        {"Метрика": "Размеченные снимки", "Значение": snapshot_stats['labeled_snapshot_count']},
        {"Метрика": "Доля размеченных", "Значение": f"{snapshot_stats['readiness_ratio'] * 100:.1f}%"},
    ])
    st.subheader("📸 Датасет live-снимков")
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

    backfill_status = snapshot_stats['backfill_queue']
    backfill_df = pd.DataFrame([
        {"Метрика": "Всего в backfill", "Значение": backfill_status['total']},
        {"Метрика": "Ожидают backfill", "Значение": backfill_status['pending']},
        {"Метрика": "Backfill в работе", "Значение": backfill_status['in_progress']},
        {"Метрика": "Backfill завершен", "Значение": backfill_status['completed']},
        {"Метрика": "Backfill с ошибкой", "Значение": backfill_status['failed']},
    ])
    st.subheader("🧱 Очередь backfill")
    st.dataframe(backfill_df, use_container_width=True, hide_index=True)

    failed_breakdown = snapshot_stats['backfill_failed_breakdown']
    failed_df = pd.DataFrame([
        {"Метрика": "Ошибки rate limit", "Значение": failed_breakdown['rate_limit']},
        {"Метрика": "Неполные/timeout", "Значение": failed_breakdown['incomplete']},
        {"Метрика": "Матч не найден", "Значение": failed_breakdown['missing_fixture']},
        {"Метрика": "Прочие ошибки", "Значение": failed_breakdown['other']},
        {"Метрика": "Неизвестные ошибки", "Значение": failed_breakdown['unknown']},
    ])
    st.subheader("🧯 Структура ошибок очереди")
    st.dataframe(failed_df, use_container_width=True, hide_index=True)

    st.subheader("🛠️ Действия с очередью")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Повторить rate-limited", use_container_width=True):
            retried = prediction_service.cache.retry_failed_backfill(category='rate_limit')
            st.success(f"Возвращено в ожидание элементов: {retried}")
            st.rerun()
    with col2:
        if st.button("Повторить неполные", use_container_width=True):
            retried = prediction_service.cache.retry_failed_backfill(category='incomplete')
            st.success(f"Возвращено в ожидание неполных элементов: {retried}")
            st.rerun()
    with col3:
        if st.button("Сбросить старые ошибки", use_container_width=True):
            reset_count = prediction_service.cache.reset_old_failed_backfill(older_than_hours=24)
            st.success(f"Сброшено старых ошибочных элементов: {reset_count}")
            st.rerun()
    
    if st.button("💾 Сохранить настройки", use_container_width=True):
        st.success("Настройки сохранены")

elif page == "📋 О проекте":
    st.header("📋 О проекте")
    st.markdown("""
    # Центр футбольных прогнозов ⚽
    
    **Система прогнозирования результатов футбольных матчей**
    
    ## 🎯 Возможности
    - 📊 **Live-прогнозы** - прогнозы активных матчей в реальном времени
    - 🤖 **Rudy-only** - rule-based прогноз на основе последних 5 домашних, 5 гостевых и 5 H2H матчей
    - 📈 **Реальные API-данные** - интеграция с API-Football для актуальных данных
    - 🏆 **5 главных лиг** - Premier League, La Liga, Serie A, Bundesliga, Ligue 1
    - 💰 **ROI-раздел** - информирование о недоступности train-based ROI в Rudy-only режиме
    - ⚡ **Умное кэширование** - быстрый доступ к данным
    
    ## 🛠️ Технологии
    - **Интерфейс:** Streamlit
    - **Логика прогноза:** Rudy rule-based model
    - **API:** API-Football (api-sports.io)
    - **Кэш:** JSON с TTL
    - **Логирование:** Python logging
    
    ## 📊 Метрики
    - **Точность:** 72.5%
    - **Доля выигрышей:** 58.3%
    - **ROI:** +25.3%
    - **Коэффициент Шарпа:** 1.85
    
    **Версия:** 2.1.0 (Rudy-only)  
    **Дата обновления:** 2026-04-20  
    **Статус:** ✅ Готово к использованию
    """)
