from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError, ReadTimeout, Timeout

from nba_api.stats.endpoints import (
    BoxScoreAdvancedV3,
    BoxScoreDefensiveV2,
    BoxScoreFourFactorsV3,
    BoxScoreHustleV2,
    BoxScoreMatchupsV3,
    BoxScoreMiscV3,
    BoxScorePlayerTrackV3,
    BoxScoreScoringV3,
    BoxScoreUsageV3,
    LeagueGameFinder,
    ScheduleLeagueV2,
)
from nba_api.stats.library.http import NBAStatsHTTP, STATS_HEADERS


RETRYABLE_EXCEPTIONS = (ConnectionError, ReadTimeout, Timeout)
REQUEST_HEADERS = STATS_HEADERS.copy()
SOURCE_COLUMNS = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
]


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    endpoint_cls: type
    table_names: tuple[str, ...]
    min_season: str
    grain: str


ENDPOINT_SPECS: dict[str, EndpointSpec] = {
    "advanced_v3": EndpointSpec(
        name="advanced_v3",
        endpoint_cls=BoxScoreAdvancedV3,
        table_names=("advanced_v3_player", "advanced_v3_team"),
        min_season="2008-09",
        grain="aligned",
    ),
    "four_factors_v3": EndpointSpec(
        name="four_factors_v3",
        endpoint_cls=BoxScoreFourFactorsV3,
        table_names=("four_factors_v3_player", "four_factors_v3_team"),
        min_season="2008-09",
        grain="aligned",
    ),
    "usage_v3": EndpointSpec(
        name="usage_v3",
        endpoint_cls=BoxScoreUsageV3,
        table_names=("usage_v3_player", "usage_v3_team"),
        min_season="2008-09",
        grain="aligned",
    ),
    "scoring_v3": EndpointSpec(
        name="scoring_v3",
        endpoint_cls=BoxScoreScoringV3,
        table_names=("scoring_v3_player", "scoring_v3_team"),
        min_season="2008-09",
        grain="aligned",
    ),
    "misc_v3": EndpointSpec(
        name="misc_v3",
        endpoint_cls=BoxScoreMiscV3,
        table_names=("misc_v3_player", "misc_v3_team"),
        min_season="2008-09",
        grain="aligned",
    ),
    "playertrack_v3": EndpointSpec(
        name="playertrack_v3",
        endpoint_cls=BoxScorePlayerTrackV3,
        table_names=("playertrack_v3_player", "playertrack_v3_team"),
        min_season="2013-14",
        grain="aligned",
    ),
    "hustle_v2": EndpointSpec(
        name="hustle_v2",
        endpoint_cls=BoxScoreHustleV2,
        table_names=("hustle_v2_player", "hustle_v2_team"),
        min_season="2016-17",
        grain="aligned",
    ),
    "defensive_v2": EndpointSpec(
        name="defensive_v2",
        endpoint_cls=BoxScoreDefensiveV2,
        table_names=("defensive_v2_player", "defensive_v2_team"),
        min_season="2017-18",
        grain="special",
    ),
    "matchups_v3": EndpointSpec(
        name="matchups_v3",
        endpoint_cls=BoxScoreMatchupsV3,
        table_names=("matchups_v3",),
        min_season="2017-18",
        grain="special",
    ),
}

DEFAULT_ENDPOINTS = tuple(ENDPOINT_SPECS.keys())


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def season_range(start_season: str, end_season: str) -> list[str]:
    start_year = season_start_year(start_season)
    end_year = season_start_year(end_season)
    if end_year < start_year:
        raise ValueError("end season must not be earlier than start season")
    seasons = []
    for year in range(start_year, end_year + 1):
        seasons.append(f"{year}-{str((year + 1) % 100).zfill(2)}")
    return seasons


def chunked(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


class RawNBADataFetcher:
    def __init__(
        self,
        seasons: list[str],
        output_dir: Path,
        endpoint_names: Sequence[str],
        max_workers: int,
        max_retries: int,
        timeout: int,
        games_per_batch: int,
        limit_games: int | None,
        retry_failed_only: bool,
    ) -> None:
        self.seasons = seasons
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.games_per_batch = games_per_batch
        self.limit_games = limit_games
        self.retry_failed_only = retry_failed_only
        self.request_headers = REQUEST_HEADERS.copy()
        self.endpoint_specs = [ENDPOINT_SPECS[name] for name in endpoint_names]
        self.endpoint_specs_by_name = {spec.name: spec for spec in self.endpoint_specs}

        session = requests.Session()
        pool_size = max(4, self.max_workers * 2)
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        NBAStatsHTTP.set_session(session)

    def run(self) -> None:
        for season in self.seasons:
            self.fetch_season(season)

    def fetch_season(self, season: str) -> None:
        logging.info("Starting season %s", season)
        season_dir = self.output_dir / f"season={season}"
        source_dir = season_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        catalog_df, regular_df, playoff_df, schedule_df = self.build_game_catalog(season)
        self._write_frame(regular_df, source_dir / "league_game_finder_regular.parquet")
        self._write_frame(playoff_df, source_dir / "league_game_finder_playoffs.parquet")
        self._write_frame(schedule_df, source_dir / "schedule_league_v2.parquet")
        self._write_frame(catalog_df, season_dir / "game_catalog.parquet")
        excluded_df = catalog_df[~catalog_df["include_for_fetch"]].copy()
        if not excluded_df.empty:
            self._write_frame(excluded_df, season_dir / "excluded_games.parquet")

        included_games = catalog_df[catalog_df["include_for_fetch"]].copy()
        included_games.sort_values(["game_date", "game_id"], inplace=True)
        if self.limit_games is not None:
            included_games = included_games.head(self.limit_games).copy()
            logging.info("Season %s limited to %d games", season, len(included_games))

        if included_games.empty:
            logging.warning("No games selected for season %s after filtering", season)
            return

        jobs_df = self._load_or_initialize_jobs(season_dir, included_games)
        active_games = self._select_active_games(jobs_df, included_games)
        self._write_frame(jobs_df, season_dir / "jobs.parquet")
        self._write_failed_jobs(jobs_df, season_dir)

        if not active_games:
            logging.info("No pending work for season %s", season)
            return

        game_lookup = {row["game_id"]: row for row in included_games.to_dict("records")}
        next_batch_id = self._next_batch_id(jobs_df)

        logging.info(
            "Season %s has %d active games across %d endpoint jobs",
            season,
            len(active_games),
            int(jobs_df["status"].isin(self._active_statuses()).sum()),
        )

        for game_batch in chunked(active_games, self.games_per_batch):
            logging.info(
                "Season %s processing batch %d with %d games",
                season,
                next_batch_id,
                len(game_batch),
            )
            batch_results: list[dict[str, Any]] = []
            table_buffers: dict[str, list[pd.DataFrame]] = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.fetch_game_jobs,
                        game_lookup[game_id],
                        endpoint_names,
                    ): game_id
                    for game_id, endpoint_names in game_batch
                }
                for future in as_completed(futures):
                    game_id = futures[future]
                    endpoint_results = future.result()
                    batch_results.extend(endpoint_results)
                    successes = sum(1 for result in endpoint_results if result["status"] == "success")
                    logging.info(
                        "Season %s game %s completed with %d/%d successes",
                        season,
                        game_id,
                        successes,
                        len(endpoint_results),
                    )
                    for result in endpoint_results:
                        for table_name, frame in result["tables"].items():
                            table_buffers.setdefault(table_name, []).append(frame)

            written_tables = self._flush_tables(season_dir, next_batch_id, table_buffers)
            jobs_df = self._apply_batch_results(jobs_df, batch_results, next_batch_id, written_tables)
            self._write_frame(jobs_df, season_dir / "jobs.parquet")
            self._write_failed_jobs(jobs_df, season_dir)
            next_batch_id += 1

        logging.info(
            "Finished season %s with %d successful endpoint jobs and %d failed endpoint jobs",
            season,
            int((jobs_df["status"] == "success").sum()),
            int(jobs_df["status"].isin({"retryable_failed", "failed_non_retryable"}).sum()),
        )

    def build_game_catalog(
        self, season: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        regular_df = self._fetch_game_finder(season, "Regular Season")
        playoff_df = self._fetch_game_finder(season, "Playoffs")
        schedule_df = self._fetch_schedule(season)

        regular_games = self._collapse_games(regular_df, season, "Regular Season")
        playoff_games = self._collapse_games(playoff_df, season, "Playoffs")
        catalog_df = pd.concat([regular_games, playoff_games], ignore_index=True)

        schedule_subset = schedule_df[
            [
                "gameId",
                "gameDate",
                "gameLabel",
                "gameSubLabel",
                "gameSubtype",
                "weekName",
                "seriesText",
                "ifNecessary",
                "homeTeam_teamId",
                "awayTeam_teamId",
            ]
        ].copy()
        schedule_subset.rename(
            columns={
                "gameId": "game_id",
                "gameDate": "schedule_game_date",
                "gameLabel": "schedule_game_label",
                "gameSubLabel": "schedule_game_sublabel",
                "gameSubtype": "schedule_game_subtype",
                "weekName": "schedule_week_name",
                "seriesText": "schedule_series_text",
                "ifNecessary": "schedule_if_necessary",
                "homeTeam_teamId": "schedule_home_team_id",
                "awayTeam_teamId": "schedule_away_team_id",
            },
            inplace=True,
        )
        schedule_subset["game_id"] = schedule_subset["game_id"].astype(str)
        schedule_subset.drop_duplicates(subset=["game_id"], inplace=True)

        catalog_df = catalog_df.merge(schedule_subset, on="game_id", how="left")
        catalog_df["is_playin"] = (
            catalog_df["schedule_game_label"].fillna("").str.contains("Play-In", case=False)
            | catalog_df["schedule_game_sublabel"].fillna("").str.contains(
                "Play-In", case=False
            )
        )
        catalog_df["is_cup_final"] = (
            catalog_df["schedule_game_label"].fillna("").eq("Emirates NBA Cup")
            & catalog_df["schedule_game_sublabel"].fillna("").str.contains(
                "Championship", case=False
            )
        )
        catalog_df["include_for_fetch"] = ~(
            catalog_df["is_playin"] | catalog_df["is_cup_final"]
        )
        catalog_df["game_date"] = pd.to_datetime(catalog_df["game_date"]).dt.strftime("%Y-%m-%d")
        catalog_df["schedule_game_date"] = pd.to_datetime(
            catalog_df["schedule_game_date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        catalog_df.sort_values(["game_date", "game_id", "season_type"], inplace=True)
        catalog_df.reset_index(drop=True, inplace=True)
        return catalog_df, regular_df, playoff_df, schedule_df

    def fetch_game_jobs(
        self, game: dict[str, Any], endpoint_names: Sequence[str]
    ) -> list[dict[str, Any]]:
        results = []
        for endpoint_name in endpoint_names:
            spec = self.endpoint_specs_by_name[endpoint_name]
            results.append(self.fetch_endpoint(spec, game))
        return results

    def fetch_endpoint(self, spec: EndpointSpec, game: dict[str, Any]) -> dict[str, Any]:
        total_elapsed = 0.0
        last_error = ""
        error_type = ""

        for attempt in range(1, self.max_retries + 1):
            start = time.perf_counter()
            try:
                endpoint = spec.endpoint_cls(
                    game_id=game["game_id"],
                    headers=self.request_headers,
                    timeout=self.timeout,
                )
                data_frames = endpoint.get_data_frames()
                if len(data_frames) != len(spec.table_names):
                    raise ValueError(
                        f"{spec.name} returned {len(data_frames)} tables, "
                        f"expected {len(spec.table_names)}"
                    )

                tables: dict[str, pd.DataFrame] = {}
                row_counts: dict[str, int] = {}
                for table_name, frame in zip(spec.table_names, data_frames):
                    table = frame.copy()
                    table["season"] = game["season"]
                    table["season_type"] = game["season_type"]
                    table["game_date"] = game["game_date"]
                    table["source_game_id"] = game["game_id"]
                    table["home_team_id"] = game["home_team_id"]
                    table["visitor_team_id"] = game["visitor_team_id"]
                    table["source_endpoint"] = spec.name
                    tables[table_name] = table
                    row_counts[table_name] = int(len(table))

                placeholder_reason = self._placeholder_reason(spec, tables)
                total_elapsed += time.perf_counter() - start
                if placeholder_reason:
                    return {
                        "game_id": game["game_id"],
                        "endpoint": spec.name,
                        "status": "placeholder_zero_data",
                        "attempts": attempt,
                        "elapsed_seconds": round(total_elapsed, 4),
                        "error_type": "",
                        "last_error": placeholder_reason,
                        "rows_json": json.dumps(row_counts, sort_keys=True),
                        "tables": {},
                    }

                return {
                    "game_id": game["game_id"],
                    "endpoint": spec.name,
                    "status": "success",
                    "attempts": attempt,
                    "elapsed_seconds": round(total_elapsed, 4),
                    "error_type": "",
                    "last_error": "",
                    "rows_json": json.dumps(row_counts, sort_keys=True),
                    "tables": tables,
                }
            except Exception as exc:
                total_elapsed += time.perf_counter() - start
                last_error = str(exc)
                error_type = type(exc).__name__
                source_missing_reason = self._source_missing_reason(spec, exc)
                if source_missing_reason:
                    return {
                        "game_id": game["game_id"],
                        "endpoint": spec.name,
                        "status": "skipped_source_missing",
                        "attempts": attempt,
                        "elapsed_seconds": round(total_elapsed, 4),
                        "error_type": error_type,
                        "last_error": source_missing_reason,
                        "rows_json": "",
                        "tables": {},
                    }
                retryable = isinstance(exc, RETRYABLE_EXCEPTIONS)
                if attempt < self.max_retries and retryable:
                    sleep_seconds = (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
                    time.sleep(sleep_seconds)
                    continue
                return {
                    "game_id": game["game_id"],
                    "endpoint": spec.name,
                    "status": "retryable_failed" if retryable else "failed_non_retryable",
                    "attempts": attempt,
                    "elapsed_seconds": round(total_elapsed, 4),
                    "error_type": error_type,
                    "last_error": last_error,
                    "rows_json": "",
                    "tables": {},
                }

        return {
            "game_id": game["game_id"],
            "endpoint": spec.name,
            "status": "failed_non_retryable",
            "attempts": self.max_retries,
            "elapsed_seconds": round(total_elapsed, 4),
            "error_type": error_type,
            "last_error": last_error,
            "rows_json": "",
            "tables": {},
        }

    def _fetch_game_finder(self, season: str, season_type: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = LeagueGameFinder(
                    player_or_team_abbreviation="T",
                    league_id_nullable="00",
                    season_nullable=season,
                    season_type_nullable=season_type,
                    headers=self.request_headers,
                    timeout=self.timeout,
                ).get_data_frames()[0]
                return df
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries and isinstance(exc, RETRYABLE_EXCEPTIONS):
                    time.sleep((2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("league game finder unexpectedly failed without exception")

    def _fetch_schedule(self, season: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = ScheduleLeagueV2(
                    league_id="00",
                    season=season,
                    headers=self.request_headers,
                    timeout=self.timeout,
                ).get_data_frames()[0]
                return df
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries and isinstance(exc, RETRYABLE_EXCEPTIONS):
                    time.sleep((2 ** (attempt - 1)) + random.uniform(0.0, 0.5))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("schedule fetch unexpectedly failed without exception")

    def _collapse_games(self, raw_df: pd.DataFrame, season: str, season_type: str) -> pd.DataFrame:
        df = raw_df[SOURCE_COLUMNS].copy()
        df["season"] = season
        df["season_type"] = season_type
        df["side"] = df["MATCHUP"].apply(lambda value: "home" if "vs." in value else "away")

        home_df = (
            df[df["side"] == "home"]
            .rename(
                columns={
                    "TEAM_ID": "home_team_id",
                    "PTS": "home_points",
                    "WL": "home_result",
                }
            )
            .drop(columns=["side"], errors="ignore")
        )
        away_df = (
            df[df["side"] == "away"]
            .rename(
                columns={
                    "TEAM_ID": "visitor_team_id",
                    "PTS": "visitor_points",
                    "WL": "visitor_result",
                }
            )
            .drop(columns=["side"], errors="ignore")
        )

        merged = pd.merge(
            home_df[
                [
                    "season",
                    "season_type",
                    "GAME_ID",
                    "GAME_DATE",
                    "SEASON_ID",
                    "home_team_id",
                    "home_points",
                    "home_result",
                ]
            ],
            away_df[
                [
                    "season",
                    "season_type",
                    "GAME_ID",
                    "GAME_DATE",
                    "SEASON_ID",
                    "visitor_team_id",
                    "visitor_points",
                    "visitor_result",
                ]
            ],
            on=["season", "season_type", "GAME_ID", "GAME_DATE", "SEASON_ID"],
            how="inner",
        )
        merged.rename(
            columns={
                "GAME_ID": "game_id",
                "GAME_DATE": "game_date",
                "SEASON_ID": "season_id",
            },
            inplace=True,
        )
        merged["game_id"] = merged["game_id"].astype(str)
        merged["result"] = merged["home_result"].fillna(
            merged.apply(
                lambda row: "W" if row["home_points"] > row["visitor_points"] else "L",
                axis=1,
            )
        )
        merged.drop(columns=["home_result", "visitor_result"], inplace=True, errors="ignore")
        merged.drop_duplicates(subset=["game_id"], inplace=True)
        return merged

    def _load_or_initialize_jobs(self, season_dir: Path, games_df: pd.DataFrame) -> pd.DataFrame:
        jobs_path = season_dir / "jobs.parquet"
        base_rows: list[dict[str, Any]] = []
        season = str(games_df.iloc[0]["season"])

        for game in games_df.to_dict("records"):
            for spec in self.endpoint_specs:
                supported = season_start_year(season) >= season_start_year(spec.min_season)
                base_rows.append(
                    {
                        "season": season,
                        "game_id": game["game_id"],
                        "game_date": game["game_date"],
                        "season_type": game["season_type"],
                        "endpoint": spec.name,
                        "grain": spec.grain,
                        "status": "pending" if supported else "skipped_unsupported",
                        "attempts": 0,
                        "elapsed_seconds": 0.0,
                        "error_type": "",
                        "last_error": "",
                        "rows_json": "",
                        "updated_at": "",
                        "output_batch_id": pd.NA,
                        "skip_reason": ""
                        if supported
                        else f"supported from {spec.min_season}",
                    }
                )

        jobs_df = pd.DataFrame(base_rows)
        if not jobs_path.exists():
            return jobs_df

        existing = pd.read_parquet(jobs_path)
        if existing.empty:
            return jobs_df
        existing = existing.sort_values("updated_at").drop_duplicates(
            subset=["game_id", "endpoint"], keep="last"
        )
        merge_columns = [
            "status",
            "attempts",
            "elapsed_seconds",
            "error_type",
            "last_error",
            "rows_json",
            "updated_at",
            "output_batch_id",
            "skip_reason",
        ]
        existing = existing[["game_id", "endpoint"] + merge_columns]
        jobs_df = jobs_df.merge(
            existing,
            on=["game_id", "endpoint"],
            how="left",
            suffixes=("", "_existing"),
        )
        for column in merge_columns:
            existing_column = f"{column}_existing"
            jobs_df[column] = jobs_df[existing_column].where(
                jobs_df[existing_column].notna(), jobs_df[column]
            )
            jobs_df.drop(columns=[existing_column], inplace=True)
        return jobs_df

    def _select_active_games(
        self, jobs_df: pd.DataFrame, games_df: pd.DataFrame
    ) -> list[tuple[str, list[str]]]:
        active_jobs = jobs_df[jobs_df["status"].isin(self._active_statuses())].copy()
        if active_jobs.empty:
            return []

        if self.retry_failed_only:
            active_jobs = active_jobs[active_jobs["status"] != "pending"].copy()
            if active_jobs.empty:
                return []

        active_jobs.sort_values(["game_date", "game_id", "endpoint"], inplace=True)
        grouped = active_jobs.groupby("game_id")["endpoint"].apply(list).to_dict()
        ordered_games = games_df[games_df["game_id"].isin(grouped.keys())].copy()
        ordered_games.sort_values(["game_date", "game_id"], inplace=True)
        return [(game_id, grouped[game_id]) for game_id in ordered_games["game_id"].tolist()]

    def _flush_tables(
        self, season_dir: Path, batch_id: int, table_buffers: dict[str, list[pd.DataFrame]]
    ) -> list[str]:
        written_tables: list[str] = []
        for table_name, frames in table_buffers.items():
            if not frames:
                continue
            table_df = pd.concat(frames, ignore_index=True)
            endpoint_name, entity = self._split_table_name(table_name)
            table_dir = season_dir / f"endpoint={endpoint_name}" / f"entity={entity}"
            table_dir.mkdir(parents=True, exist_ok=True)
            part_path = table_dir / f"part-{batch_id:05d}.parquet"
            table_df.to_parquet(part_path, index=False)
            written_tables.append(table_name)
        return written_tables

    def _apply_batch_results(
        self,
        jobs_df: pd.DataFrame,
        batch_results: list[dict[str, Any]],
        batch_id: int,
        written_tables: list[str],
    ) -> pd.DataFrame:
        updated_at = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        table_set = set(written_tables)
        for result in batch_results:
            mask = (jobs_df["game_id"] == result["game_id"]) & (jobs_df["endpoint"] == result["endpoint"])
            jobs_df.loc[mask, "status"] = result["status"]
            jobs_df.loc[mask, "attempts"] = result["attempts"]
            jobs_df.loc[mask, "elapsed_seconds"] = result["elapsed_seconds"]
            jobs_df.loc[mask, "error_type"] = result["error_type"]
            jobs_df.loc[mask, "last_error"] = result["last_error"]
            jobs_df.loc[mask, "rows_json"] = result["rows_json"]
            jobs_df.loc[mask, "updated_at"] = updated_at

            endpoint_tables = self.endpoint_specs_by_name[result["endpoint"]].table_names
            has_output = result["status"] == "success" and any(
                table_name in table_set for table_name in endpoint_tables
            )
            jobs_df.loc[mask, "output_batch_id"] = batch_id if has_output else pd.NA
        return jobs_df

    def _write_failed_jobs(self, jobs_df: pd.DataFrame, season_dir: Path) -> None:
        failed_jobs = jobs_df[
            jobs_df["status"].isin({"retryable_failed", "failed_non_retryable", "placeholder_zero_data"})
        ].copy()
        failed_path = season_dir / "failed_jobs.parquet"
        if failed_jobs.empty:
            if failed_path.exists():
                failed_path.unlink()
            return
        self._write_frame(failed_jobs, failed_path)

    def _next_batch_id(self, jobs_df: pd.DataFrame) -> int:
        batch_values = pd.to_numeric(jobs_df["output_batch_id"], errors="coerce").dropna()
        if batch_values.empty:
            return 1
        return int(batch_values.max()) + 1

    def _placeholder_reason(self, spec: EndpointSpec, tables: dict[str, pd.DataFrame]) -> str:
        if spec.name != "hustle_v2":
            return ""
        player_table = tables.get("hustle_v2_player")
        if player_table is None or player_table.empty:
            return ""
        if "teamId" not in player_table.columns:
            return ""
        if len(player_table) <= 2 and player_table["teamId"].fillna(0).eq(0).all():
            return "placeholder zero rows returned by hustle endpoint"
        return ""

    def _source_missing_reason(self, spec: EndpointSpec, exc: Exception) -> str:
        if spec.name == "defensive_v2" and isinstance(exc, AttributeError):
            if "'NoneType' object has no attribute 'get'" in str(exc):
                return "source returned no defensive data for this game"
        if spec.name == "matchups_v3" and isinstance(exc, IndexError):
            if "list index out of range" in str(exc):
                return "source returned no matchup data for this game"
        return ""

    def _active_statuses(self) -> set[str]:
        return {"pending", "retryable_failed", "failed_non_retryable"}

    def _split_table_name(self, table_name: str) -> tuple[str, str]:
        if table_name == "matchups_v3":
            return "matchups_v3", "matchup"
        endpoint_name, entity = table_name.rsplit("_", 1)
        return endpoint_name, entity

    def _write_frame(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw NBA data season by season.")
    parser.add_argument("--seasons", nargs="*", help="Explicit seasons like 2018-19 2019-20")
    parser.add_argument("--from-season", help="Range start season, e.g. 2008-09")
    parser.add_argument("--to-season", help="Range end season, e.g. 2024-25")
    parser.add_argument("--output-dir", default="data/raw", help="Raw data output directory")
    parser.add_argument(
        "--endpoints",
        nargs="*",
        default=list(DEFAULT_ENDPOINTS),
        choices=sorted(ENDPOINT_SPECS.keys()),
        help="Endpoints to fetch",
    )
    parser.add_argument("--max-workers", type=int, default=3, help="Concurrent game workers")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per endpoint call")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout seconds")
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=10,
        help="Games to flush together into parquet part files",
    )
    parser.add_argument("--limit-games", type=int, default=None, help="Optional game cap per season")
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="Only rerun jobs that previously failed",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def resolve_seasons(args: argparse.Namespace) -> list[str]:
    if args.seasons:
        return list(dict.fromkeys(args.seasons))
    if args.from_season and args.to_season:
        return season_range(args.from_season, args.to_season)
    raise ValueError("Provide either --seasons or both --from-season and --to-season.")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    seasons = resolve_seasons(args)
    fetcher = RawNBADataFetcher(
        seasons=seasons,
        output_dir=Path(args.output_dir),
        endpoint_names=args.endpoints,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        timeout=args.timeout,
        games_per_batch=args.games_per_batch,
        limit_games=args.limit_games,
        retry_failed_only=args.retry_failed_only,
    )
    fetcher.run()


if __name__ == "__main__":
    main()
