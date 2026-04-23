"""Автоматическое обновление данных по расписанию (Rudy-only, без обучения)."""
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import (
    TRACKED_LEAGUES,
    SCHEDULE_TIME_1,
    SCHEDULE_TIME_2,
    SCHEDULER_ENABLED,
    DETAIL_BACKFILL_ENABLED,
    DETAIL_BACKFILL_MAX_FIXTURES_PER_UPDATE,
    DETAIL_BACKFILL_INCLUDE_STATISTICS,
    DETAIL_BACKFILL_INCLUDE_ODDS,
    BACKFILL_QUEUE_MAX_ATTEMPTS,
    LIVE_SNAPSHOTS_ENABLED,
    LIVE_SNAPSHOT_MIN_MINUTE,
    LIVE_SNAPSHOT_MAX_MINUTE,
    LIVE_SNAPSHOT_POLL_MINUTES,
    DASHBOARD_PREDICTIONS_POLL_MINUTES,
    get_current_football_season,
)
from data.api_client import FootballAPIClient
from data.cache_manager import CacheManager
from data.data_service import DataService
from prediction_service import PredictionService

logger = logging.getLogger(__name__)


class DataUpdateScheduler:
    """Планировщик автоматического обновления данных без автопереобучения моделей."""

    def __init__(self):
        self.scheduler = BackgroundScheduler(job_defaults={'coalesce': True, 'max_instances': 1})
        self.api_client = FootballAPIClient()
        self.cache = CacheManager()
        self.data_service = DataService()
        self.prediction_service = None
        self.is_running = False

    def start(self):
        """Запустить планировщик."""
        if not SCHEDULER_ENABLED:
            logger.warning("Scheduler is disabled in settings")
            return

        try:
            hour1, min1 = map(int, SCHEDULE_TIME_1.split(':'))
            self.scheduler.add_job(
                self._update_data,
                CronTrigger(hour=hour1, minute=min1),
                id='update_data_1',
                name='Data update at schedule #1',
            )

            hour2, min2 = map(int, SCHEDULE_TIME_2.split(':'))
            self.scheduler.add_job(
                self._update_data,
                CronTrigger(hour=hour2, minute=min2),
                id='update_data_2',
                name='Data update at schedule #2',
            )

            self.scheduler.add_job(
                self._cleanup_cache,
                CronTrigger(hour=1, minute=0),
                id='cleanup_cache',
                name='Cache cleanup',
            )

            if LIVE_SNAPSHOTS_ENABLED:
                self.scheduler.add_job(
                    self._collect_live_snapshots,
                    IntervalTrigger(minutes=LIVE_SNAPSHOT_POLL_MINUTES),
                    id='collect_live_snapshots',
                    name='Live snapshot collection',
                )

            self.scheduler.add_job(
                self._prepare_dashboard_predictions,
                IntervalTrigger(minutes=DASHBOARD_PREDICTIONS_POLL_MINUTES),
                id='prepare_dashboard_predictions',
                name='Dashboard predictions cache refresh',
                next_run_time=datetime.now(),
            )

            self.scheduler.start()
            self.is_running = True
            logger.info("Data update scheduler started (Rudy-only mode)")
        except Exception as e:
            logger.error("Error starting scheduler: %s", e)

    def stop(self):
        """Остановить планировщик."""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Data update scheduler stopped")

    def _update_data(self):
        """Загрузить новые данные из API."""
        logger.info("Starting scheduled data update...")

        try:
            current_season = get_current_football_season()

            for league_id, league_name in TRACKED_LEAGUES.items():
                try:
                    fixtures = self.api_client.get_fixtures(league=league_id, season=current_season)
                    if self.api_client.last_error_reason == 'rate_limit':
                        logger.warning(
                            "Stopping scheduled data update after rate limit on fixtures for %s",
                            league_name,
                        )
                        break

                    detail_budget = DETAIL_BACKFILL_MAX_FIXTURES_PER_UPDATE
                    enriched_fixtures = []
                    for fixture in fixtures:
                        if (
                            DETAIL_BACKFILL_ENABLED
                            and detail_budget > 0
                            and fixture.get('fixture', {}).get('status', {}).get('short') == 'FT'
                        ):
                            fixture = self.api_client.enrich_fixture_details(
                                fixture,
                                include_statistics=DETAIL_BACKFILL_INCLUDE_STATISTICS,
                                include_odds=DETAIL_BACKFILL_INCLUDE_ODDS,
                            )
                            if self.api_client.last_error_reason == 'rate_limit':
                                logger.warning(
                                    "Rate limit reached during detail enrichment for %s; keeping remaining fixtures without extra API calls",
                                    league_name,
                                )
                                detail_budget = 0
                            detail_budget -= 1
                        enriched_fixtures.append(fixture)

                    for fixture in enriched_fixtures:
                        self.cache.save_fixture(fixture)
                        if fixture.get('fixture', {}).get('status', {}).get('short') == 'FT':
                            self.cache.enqueue_fixture_for_backfill(fixture)

                    standings = self.api_client.get_standings(league_id, current_season)
                    if self.api_client.last_error_reason == 'rate_limit':
                        logger.warning(
                            "Stopping scheduled data update before standings refresh for %s due to rate limit",
                            league_name,
                        )
                        break
                    if standings:
                        self.cache.set(
                            f"standings_{league_id}_{current_season}",
                            standings,
                            ttl=86400,
                        )

                    logger.info("Updated %s: %s fixtures", league_name, len(enriched_fixtures))
                    self.cache.log_api_call("fixtures", {"league": league_id}, "success")
                except Exception as e:
                    logger.error("Error updating %s: %s", league_name, e)
                    self.cache.log_api_call("fixtures", {"league": league_id}, "error")

            logger.info("Data update completed successfully")
            self._process_backfill_queue_batch()
        except Exception as e:
            logger.error("Critical error during data update: %s", e)

    def _cleanup_cache(self):
        """Очистить устаревшие записи кэша."""
        try:
            count = self.cache.cleanup_expired()
            logger.info("Cache cleanup completed: %s expired records deleted", count)
        except Exception as e:
            logger.error("Error during cache cleanup: %s", e)

    def _process_backfill_queue_batch(self):
        """Обработать батч persistent backfill queue."""
        try:
            if self.api_client.last_error_reason == 'rate_limit':
                logger.warning("Skipping backfill queue batch because API is rate limited")
                return

            queue_batch = self.cache.claim_backfill_batch(
                batch_size=DETAIL_BACKFILL_MAX_FIXTURES_PER_UPDATE,
                max_attempts=BACKFILL_QUEUE_MAX_ATTEMPTS,
            )
            if not queue_batch:
                return

            processed = 0
            for queue_item in queue_batch:
                fixture_id = queue_item['fixture_id']
                fixture_data = self.cache.get_fixture_by_id(fixture_id)
                if fixture_data is None:
                    self.cache.mark_backfill_failed(fixture_id, 'fixture_not_found')
                    continue

                enriched_fixture = self.api_client.enrich_fixture_details(
                    fixture_data,
                    include_statistics=bool(queue_item.get('requested_statistics')),
                    include_odds=bool(queue_item.get('requested_odds')),
                )
                if self.api_client.last_error_reason == 'rate_limit':
                    self.cache.mark_backfill_failed(fixture_id, 'rate_limit')
                    logger.warning("Stopping backfill queue batch after rate limit on fixture %s", fixture_id)
                    break
                self.cache.save_fixture(enriched_fixture)

                has_statistics = bool(
                    enriched_fixture.get('statistics', {}).get('home')
                    or enriched_fixture.get('statistics', {}).get('away')
                )
                current_odds = enriched_fixture.get('odds', {})
                has_odds = all(current_odds.get(key) is not None for key in ('1', 'X', '2'))
                detail_errors = enriched_fixture.get('detail_errors', {})
                if (
                    (not queue_item.get('requested_statistics') or has_statistics)
                    and (not queue_item.get('requested_odds') or has_odds)
                ):
                    self.cache.mark_backfill_completed(fixture_id)
                else:
                    error_reason = (
                        detail_errors.get('statistics')
                        or detail_errors.get('odds')
                        or 'details_incomplete_or_rate_limited'
                    )
                    self.cache.mark_backfill_failed(fixture_id, error_reason)
                processed += 1

            if processed:
                logger.info("Processed %s backfill queue items", processed)
        except Exception as e:
            logger.error("Error processing backfill queue: %s", e)

    def _collect_live_snapshots(self):
        """Сохранить live snapshots для активных матчей без участия UI."""
        if not LIVE_SNAPSHOTS_ENABLED:
            return

        try:
            live_fixtures = self.api_client.get_live_fixtures()
            saved_count = 0
            for fixture in live_fixtures:
                elapsed = fixture.get('fixture', {}).get('status', {}).get('elapsed')
                if elapsed is None:
                    continue
                if elapsed < LIVE_SNAPSHOT_MIN_MINUTE or elapsed > LIVE_SNAPSHOT_MAX_MINUTE:
                    continue

                enriched_fixture = self.api_client.enrich_fixture_details(
                    fixture,
                    include_statistics=True,
                    include_odds=True,
                )
                if self.cache.save_live_snapshot(enriched_fixture):
                    saved_count += 1

            if saved_count:
                logger.info("Collected %s live snapshots", saved_count)
        except Exception as e:
            logger.error("Error collecting live snapshots: %s", e)

    def _prepare_dashboard_predictions(self):
        """Фоново подготовить прогнозы для Dashboard и сохранить их в cache."""
        started_at = datetime.now().isoformat()
        self.cache.set(
            PredictionService.DASHBOARD_STATE_CACHE_KEY,
            {
                'status': 'running',
                'started_at': started_at,
                'finished_at': None,
                'last_error': None,
                'row_count': 0,
            },
            ttl=86400,
        )
        try:
            if self.prediction_service is None:
                self.prediction_service = PredictionService()

            fixtures = self.data_service.get_today_fixtures(use_cache=True)
            payload = self.prediction_service.refresh_dashboard_prediction_cache(fixtures)
            self.cache.set(
                PredictionService.DASHBOARD_STATE_CACHE_KEY,
                {
                    'status': 'ready',
                    'started_at': started_at,
                    'finished_at': datetime.now().isoformat(),
                    'last_error': None,
                    'row_count': int(payload.get('row_count', 0)),
                },
                ttl=86400,
            )
            logger.info("Prepared %s dashboard predictions in background", payload.get('row_count', 0))
        except Exception as e:
            self.cache.set(
                PredictionService.DASHBOARD_STATE_CACHE_KEY,
                {
                    'status': 'error',
                    'started_at': started_at,
                    'finished_at': datetime.now().isoformat(),
                    'last_error': str(e),
                    'row_count': 0,
                },
                ttl=86400,
            )
            logger.error("Error preparing dashboard predictions: %s", e)

    def force_update(self):
        """Принудительно обновить данные."""
        logger.info("Force data update initiated")
        self._update_data()
        self._prepare_dashboard_predictions()

    def get_status(self) -> dict:
        """Получить статус планировщика."""
        jobs = []
        if self.is_running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': str(job.next_run_time),
                })

        return {
            'is_running': self.is_running,
            'jobs': jobs,
            'cache_stats': self.cache.get_stats(),
        }
