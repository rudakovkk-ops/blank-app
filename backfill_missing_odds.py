"""Mass backfill of missing market odds for cached finished fixtures."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import LOGS_DIR, TRACKED_LEAGUES, TRAINING_DATA_SEASONS, get_current_football_season
from data.api_client import FootballAPIClient
from data.cache_manager import CacheManager


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _fixture_has_full_odds(fixture: dict) -> bool:
    odds = fixture.get("odds", {})
    return all(odds.get(key) is not None for key in ("1", "X", "2"))


def _active_seasons(history_depth: int) -> list[int]:
    current_season = get_current_football_season()
    depth = max(1, int(history_depth))
    return [current_season - offset for offset in range(depth)]


def collect_cached_finished_fixtures(cache: CacheManager) -> list[dict]:
    fixtures: list[dict] = []
    seasons = _active_seasons(TRAINING_DATA_SEASONS)

    for league_id in TRACKED_LEAGUES.keys():
        for season in seasons:
            fixtures.extend(cache.get_fixtures_by_league_season(league_id, season))

    finished = [
        fixture
        for fixture in fixtures
        if fixture.get("fixture", {}).get("status", {}).get("short") == "FT"
    ]

    def fixture_dt(value: dict) -> datetime:
        raw_date = value.get("fixture", {}).get("date") or value.get("date")
        try:
            return datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    finished.sort(key=fixture_dt, reverse=True)
    return finished


def run_backfill(max_fixtures: int, max_age_days: int | None = None, save_report: bool = True) -> dict:
    cache = CacheManager()
    api = FootballAPIClient()

    finished = collect_cached_finished_fixtures(cache)
    total_finished = len(finished)

    with_odds_before = sum(1 for fixture in finished if _fixture_has_full_odds(fixture))
    missing = [fixture for fixture in finished if not _fixture_has_full_odds(fixture)]

    if max_age_days is not None and max_age_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        def is_recent(fixture: dict) -> bool:
            raw_date = fixture.get("fixture", {}).get("date") or fixture.get("date")
            try:
                fixture_date = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
            except Exception:
                return False
            return fixture_date >= cutoff

        missing = [fixture for fixture in missing if is_recent(fixture)]

    enriched_attempts = 0
    enriched_success = 0
    unchanged = 0
    api_errors: dict[str, int] = {}

    target = missing[: max(0, int(max_fixtures))] if max_fixtures > 0 else []

    for index, fixture in enumerate(target, start=1):
        fixture_id = fixture.get("fixture", {}).get("id")
        enriched_attempts += 1

        enriched = api.enrich_fixture_details(
            fixture,
            include_statistics=False,
            include_odds=True,
            force=True,
        )

        last_error = api.last_error_reason
        if last_error:
            api_errors[last_error] = api_errors.get(last_error, 0) + 1
            if last_error == "rate_limit":
                logger.warning("Rate limit reached on fixture_id=%s, stopping current run", fixture_id)
                break

        has_odds_now = _fixture_has_full_odds(enriched)
        if has_odds_now:
            if cache.save_fixture(enriched):
                enriched_success += 1
            else:
                api_errors["save_error"] = api_errors.get("save_error", 0) + 1
        else:
            unchanged += 1
            cache.save_fixture(enriched)

        if index % 10 == 0 or index == len(target):
            logger.info(
                "Progress: %s/%s attempts, success=%s, unchanged=%s, errors=%s",
                index,
                len(target),
                enriched_success,
                unchanged,
                sum(api_errors.values()),
            )

    refreshed = collect_cached_finished_fixtures(cache)
    with_odds_after = sum(1 for fixture in refreshed if _fixture_has_full_odds(fixture))

    report = {
        "created_at": datetime.now().isoformat(),
        "total_finished_fixtures": total_finished,
        "with_full_odds_before": with_odds_before,
        "with_full_odds_after": with_odds_after,
        "coverage_before": (with_odds_before / total_finished) if total_finished else 0.0,
        "coverage_after": (with_odds_after / total_finished) if total_finished else 0.0,
        "coverage_delta": ((with_odds_after - with_odds_before) / total_finished) if total_finished else 0.0,
        "attempted_fixtures": enriched_attempts,
        "successfully_enriched": enriched_success,
        "unchanged_after_enrichment": unchanged,
        "api_errors": api_errors,
        "NOTE": "Backfill обновляет odds и статистику в кэше для более точного Rudy-инференса.",
    }

    if save_report:
        LOGS_DIR.mkdir(exist_ok=True)
        report_path = LOGS_DIR / "odds_backfill_run_latest.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill missing odds for cached finished fixtures")
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=200,
        help="Maximum number of missing-odds fixtures to process in one run",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=180,
        help="Only attempt fixtures not older than this many days",
    )
    args = parser.parse_args()

    report = run_backfill(
        max_fixtures=args.max_fixtures,
        max_age_days=args.max_age_days,
        save_report=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
