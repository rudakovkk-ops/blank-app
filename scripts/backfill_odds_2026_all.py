from __future__ import annotations

import json
from datetime import datetime

from config.settings import LOGS_DIR, TRACKED_LEAGUES
from data.api_client import FootballAPIClient
from data.cache_manager import CacheManager


def has_1x2(odds: dict) -> bool:
    return odds.get("1") is not None and odds.get("X") is not None and odds.get("2") is not None


def has_totals(odds: dict) -> bool:
    return odds.get("over_2_5") is not None and odds.get("under_2_5") is not None


def main() -> None:
    cache = CacheManager()
    api = FootballAPIClient()

    report_path = LOGS_DIR / "odds_backfill_2026_latest.json"
    LOGS_DIR.mkdir(exist_ok=True)

    def write_report(payload: dict) -> None:
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fixtures = []
    for league_id in TRACKED_LEAGUES.keys():
        for season in (2026, 2025, 2024, 2023):
            fixtures.extend(cache.get_fixtures_by_league_season(league_id, season))

    seen = set()
    unique = []
    for fixture in fixtures:
        fixture_id = fixture.get("fixture", {}).get("id")
        if fixture_id is None or fixture_id in seen:
            continue
        seen.add(fixture_id)
        unique.append(fixture)

    rows_2026 = []
    for fixture in unique:
        raw_date = fixture.get("fixture", {}).get("date") or fixture.get("date")
        try:
            dt = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.year == 2026:
            rows_2026.append(fixture)

    pending = []
    for fixture in rows_2026:
        odds = fixture.get("odds", {}) or {}
        if not (has_1x2(odds) and has_totals(odds)):
            pending.append(fixture)

    payload = {
        "created_at": datetime.now().isoformat(),
        "status": "running",
        "year": 2026,
        "fixtures_2026_total": len(rows_2026),
        "target_pending": len(pending),
        "attempts": 0,
        "success": 0,
        "unchanged": 0,
        "errors": {},
        "progress_pct": 0.0,
        "remaining": len(pending),
    }
    write_report(payload)
    print(f"[START] total_2026={len(rows_2026)} pending={len(pending)}", flush=True)

    attempts = 0
    success = 0
    unchanged = 0
    errors: dict[str, int] = {}

    for idx, fixture in enumerate(pending, start=1):
        attempts += 1
        fixture_id = fixture.get("fixture", {}).get("id")

        enriched = api.enrich_fixture_details(
            fixture,
            include_statistics=False,
            include_odds=True,
            force=True,
        )

        last_error = api.last_error_reason
        if last_error:
            errors[last_error] = errors.get(last_error, 0) + 1

        odds = enriched.get("odds", {}) or {}
        ok = has_1x2(odds) and has_totals(odds)
        if ok:
            if cache.save_fixture(enriched):
                success += 1
            else:
                errors["save_error"] = errors.get("save_error", 0) + 1
        else:
            unchanged += 1
            cache.save_fixture(enriched)

        payload = {
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "year": 2026,
            "fixtures_2026_total": len(rows_2026),
            "target_pending": len(pending),
            "attempts": attempts,
            "success": success,
            "unchanged": unchanged,
            "errors": errors,
            "progress_pct": round((attempts / max(len(pending), 1)) * 100, 2),
            "remaining": max(len(pending) - attempts, 0),
            "last_fixture_id": fixture_id,
        }
        write_report(payload)

        if idx % 10 == 0 or idx == len(pending):
            print(
                f"[PROGRESS] {attempts}/{len(pending)} success={success} "
                f"unchanged={unchanged} errors={sum(errors.values())}",
                flush=True,
            )

    payload = {
        "created_at": datetime.now().isoformat(),
        "status": "completed",
        "year": 2026,
        "fixtures_2026_total": len(rows_2026),
        "target_pending": len(pending),
        "attempts": attempts,
        "success": success,
        "unchanged": unchanged,
        "errors": errors,
        "progress_pct": 100.0,
        "remaining": 0,
    }
    write_report(payload)
    print("[DONE] backfill_2026 complete", flush=True)
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
