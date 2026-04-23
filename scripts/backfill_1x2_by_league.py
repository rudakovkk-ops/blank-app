from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime

from config.settings import LOGS_DIR, TRACKED_LEAGUES
from data.api_client import FootballAPIClient
from data.cache_manager import CacheManager


def has_1x2(odds: dict) -> bool:
    return odds.get("1") is not None and odds.get("X") is not None and odds.get("2") is not None


def fixture_year(fixture: dict) -> int | None:
    raw_date = fixture.get("fixture", {}).get("date") or fixture.get("date")
    if not raw_date:
        return None
    try:
        return datetime.fromisoformat(str(raw_date).replace("Z", "+00:00")).year
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="1X2 backfill by league for 2026")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--max-checks-per-league", type=int, default=40)
    parser.add_argument(
        "--only-with-data",
        action="store_true",
        help="Skip leagues with zero fixtures in cache for the selected year",
    )
    args = parser.parse_args()

    cache = CacheManager()
    api = FootballAPIClient()

    report = {
        "created_at": datetime.now().isoformat(),
        "policy": "1X2 only",
        "year": args.year,
        "max_checks_per_league": args.max_checks_per_league,
        "league_rows": [],
        "totals": defaultdict(int),
    }

    league_inputs = []
    for league_id, league_name in TRACKED_LEAGUES.items():
        fixtures = cache.get_fixtures_by_league_season(league_id, args.year)
        fixtures_by_year = [f for f in fixtures if fixture_year(f) == args.year]
        if args.only_with_data and not fixtures_by_year:
            continue
        league_inputs.append((league_id, league_name, fixtures_by_year))

    league_inputs.sort(key=lambda row: len(row[2]), reverse=True)

    for league_id, league_name, fixtures_2026 in league_inputs:

        before_missing = []
        for fixture in fixtures_2026:
            odds = fixture.get("odds") or {}
            if not has_1x2(odds):
                before_missing.append(fixture)

        stats = defaultdict(int)
        stats["league_id"] = league_id
        stats["league_name"] = league_name
        stats["fixtures_2026_total"] = len(fixtures_2026)
        stats["missing_before"] = len(before_missing)

        for fixture in before_missing[: max(args.max_checks_per_league, 0)]:
            fixture_id = fixture.get("fixture", {}).get("id")
            if fixture_id is None:
                continue

            stats["checked"] += 1
            odds_response = api.get_odds(fixture_id)
            if odds_response:
                stats["api_nonempty"] += 1
            else:
                stats["api_empty"] += 1

            enriched = api.enrich_fixture_details(
                fixture,
                include_statistics=False,
                include_odds=True,
                force=True,
            )

            odds = enriched.get("odds", {}) or {}
            if has_1x2(odds):
                if cache.save_fixture(enriched):
                    stats["saved_1x2"] += 1
                else:
                    stats["save_errors"] += 1
            else:
                stats["no_1x2_after_call"] += 1
                cache.save_fixture(enriched)

            if api.last_error_reason:
                key = f"error_{api.last_error_reason}"
                stats[key] += 1

        after_with_1x2 = 0
        refreshed = cache.get_fixtures_by_league_season(league_id, args.year)
        for fixture in refreshed:
            if fixture_year(fixture) != args.year:
                continue
            odds = fixture.get("odds") or {}
            if has_1x2(odds):
                after_with_1x2 += 1

        stats["with_1x2_after"] = after_with_1x2
        stats["with_1x2_rate_after"] = round(
            (after_with_1x2 / max(len(fixtures_2026), 1)) * 100, 2
        )

        report["league_rows"].append(dict(stats))
        for k, v in stats.items():
            if isinstance(v, int):
                report["totals"][k] += v

        print(
            f"[LEAGUE] {league_id} {league_name}: total={stats['fixtures_2026_total']} "
            f"missing_before={stats['missing_before']} checked={stats['checked']} "
            f"saved_1x2={stats['saved_1x2']} "
            f"api_empty={stats['api_empty']} no_1x2_after_call={stats['no_1x2_after_call']}",
            flush=True,
        )

    report["totals"] = dict(report["totals"])
    report_path = LOGS_DIR / f"backfill_1x2_by_league_report_{args.year}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| league_id | league_name | total_2026 | missing_before | checked | api_nonempty | api_empty | saved_1x2 | no_1x2_after_call | with_1x2_after | rate_after_% |", flush=True)
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    for row in sorted(report["league_rows"], key=lambda x: x.get("with_1x2_rate_after", 0), reverse=True):
        print(
            "| {league_id} | {league_name} | {fixtures_2026_total} | {missing_before} | {checked} | {api_nonempty} | {api_empty} | {saved_1x2} | {no_1x2_after_call} | {with_1x2_after} | {with_1x2_rate_after} |".format(
                league_id=row.get("league_id", 0),
                league_name=row.get("league_name", ""),
                fixtures_2026_total=row.get("fixtures_2026_total", 0),
                missing_before=row.get("missing_before", 0),
                checked=row.get("checked", 0),
                api_nonempty=row.get("api_nonempty", 0),
                api_empty=row.get("api_empty", 0),
                saved_1x2=row.get("saved_1x2", 0),
                no_1x2_after_call=row.get("no_1x2_after_call", 0),
                with_1x2_after=row.get("with_1x2_after", 0),
                with_1x2_rate_after=row.get("with_1x2_rate_after", 0),
            ),
            flush=True,
        )

    print(f"\n[REPORT] {report_path}", flush=True)


if __name__ == "__main__":
    main()
