from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


CORE_REQUIRED_ENDPOINTS = (
    "advanced_v3",
    "four_factors_v3",
    "usage_v3",
    "scoring_v3",
    "misc_v3",
    "playertrack_v3",
    "hustle_v2",
)
OPTIONAL_PLAYER_ENDPOINTS = ("defensive_v2",)
PLAYER_ENDPOINTS = ("advanced_v3",) + CORE_REQUIRED_ENDPOINTS[1:] + OPTIONAL_PLAYER_ENDPOINTS
TEAM_ENDPOINTS = (
    "advanced_v3",
    "four_factors_v3",
    "usage_v3",
    "scoring_v3",
    "misc_v3",
    "playertrack_v3",
    "hustle_v2",
)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw NBA data for player-set models.")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--seasons", nargs="*", help="Explicit seasons like 2023-24 2024-25")
    parser.add_argument("--top-k-players", type=int, default=10, help="Players per team set")
    parser.add_argument(
        "--min-team-history-games",
        type=int,
        default=10,
        help="Default training-eligibility threshold for prior team games",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def list_available_seasons(raw_dir: Path) -> list[str]:
    seasons = []
    for path in raw_dir.glob("season=*"):
        seasons.append(path.name.split("=", 1)[1])
    return sorted(seasons)


def camel_to_snake(name: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = text.replace("%", "pct")
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    return text.strip("_").lower()


def parse_minutes_to_float(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return float(parts[0]) + float(parts[1]) / 60.0
    return float(text)


def load_part_files(directory: Path) -> pd.DataFrame:
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)


def build_available_games(
    raw_dir: Path,
    seasons: Sequence[str],
    required_endpoints: Sequence[str],
) -> pd.DataFrame:
    jobs = []
    catalogs = []
    for season in seasons:
        season_dir = raw_dir / f"season={season}"
        jobs.append(pd.read_parquet(season_dir / "jobs.parquet"))
        catalog = pd.read_parquet(season_dir / "game_catalog.parquet")
        catalogs.append(catalog[catalog["include_for_fetch"]].copy())

    jobs_df = pd.concat(jobs, ignore_index=True)
    catalogs_df = pd.concat(catalogs, ignore_index=True)
    core_jobs = jobs_df[jobs_df["endpoint"].isin(required_endpoints)].copy()
    success_matrix = core_jobs.pivot_table(
        index=["season", "game_id"],
        columns="endpoint",
        values="status",
        aggfunc="first",
    )
    complete_games = success_matrix.index[(success_matrix == "success").all(axis=1)]
    complete_games = pd.DataFrame(complete_games.tolist(), columns=["season", "game_id"])
    catalogs_df["game_id"] = catalogs_df["game_id"].astype(str)
    return catalogs_df.merge(complete_games, on=["season", "game_id"], how="inner")


def dedupe_player_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dedupe_keys = ["game_id", "team_id", "person_id"]
    score_columns = [
        column
        for column in df.columns
        if column not in dedupe_keys and column not in {"first_name", "family_name", "player_name"}
    ]
    working = df.copy()
    working["minutes_rank"] = working["minutes_raw"].fillna("").astype(str).str.len().gt(0).astype(int)
    working["nonnull_rank"] = working[score_columns].notna().sum(axis=1)
    working.sort_values(
        dedupe_keys + ["minutes_rank", "nonnull_rank"],
        ascending=[True, True, True, False, False],
        inplace=True,
    )
    working = working.drop_duplicates(subset=dedupe_keys, keep="first")
    working.drop(columns=["minutes_rank", "nonnull_rank"], inplace=True)
    return working


def normalize_player_endpoint(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    rename_map = {
        "source_game_id": "game_id",
        "teamId": "team_id",
        "personId": "person_id",
        "firstName": "first_name",
        "familyName": "family_name",
        "position": "position",
        "comment": "comment",
        "minutes": "minutes_raw",
    }
    working.rename(columns=rename_map, inplace=True)
    working["game_id"] = working["game_id"].astype(str)
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
    working["person_id"] = pd.to_numeric(working["person_id"], errors="coerce").astype("Int64")
    working["player_name"] = (
        working["first_name"].fillna("").astype(str).str.strip()
        + " "
        + working["family_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    if "minutes_raw" in working.columns:
        working["minutes_raw"] = working["minutes_raw"].fillna("")
    else:
        working["minutes_raw"] = ""
    working["minutes_played"] = working["minutes_raw"].map(parse_minutes_to_float)

    keep_meta = {
        "season",
        "season_type",
        "game_date",
        "game_id",
        "team_id",
        "person_id",
        "first_name",
        "family_name",
        "player_name",
        "position",
        "comment",
        "minutes_raw",
        "minutes_played",
    }
    drop_columns = {
        "source_endpoint",
        "home_team_id",
        "visitor_team_id",
        "teamCity",
        "teamName",
        "teamTricode",
        "teamSlug",
        "nameI",
        "playerSlug",
        "jerseyNum",
    }
    feature_columns = []
    for column in working.columns:
        if column in keep_meta or column in drop_columns:
            continue
        if column in {"game_id", "team_id", "person_id"}:
            continue
        if camel_to_snake(column).endswith("_id"):
            continue
        feature_columns.append(column)

    for column in feature_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    rename_features = {column: f"{endpoint}__{camel_to_snake(column)}" for column in feature_columns}
    working.rename(columns=rename_features, inplace=True)
    final_columns = list(keep_meta) + list(rename_features.values())
    final_df = working[final_columns].copy()
    return dedupe_player_endpoint(final_df)


def normalize_team_endpoint(df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    rename_map = {
        "source_game_id": "game_id",
        "teamId": "team_id",
        "minutes": "minutes_raw",
    }
    working.rename(columns=rename_map, inplace=True)
    working["game_id"] = working["game_id"].astype(str)
    working["team_id"] = pd.to_numeric(working["team_id"], errors="coerce").astype("Int64")
    if "minutes_raw" in working.columns:
        working["minutes_raw"] = working["minutes_raw"].fillna("")
    else:
        working["minutes_raw"] = ""
    working["minutes_played"] = working["minutes_raw"].map(parse_minutes_to_float)

    keep_meta = {"season", "season_type", "game_date", "game_id", "team_id", "minutes_raw", "minutes_played"}
    drop_columns = {
        "source_endpoint",
        "home_team_id",
        "visitor_team_id",
        "teamCity",
        "teamName",
        "teamTricode",
        "teamSlug",
    }
    feature_columns = []
    for column in working.columns:
        if column in keep_meta or column in drop_columns:
            continue
        if column in {"game_id", "team_id"}:
            continue
        if camel_to_snake(column).endswith("_id"):
            continue
        feature_columns.append(column)

    for column in feature_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    rename_features = {column: f"{endpoint}__{camel_to_snake(column)}" for column in feature_columns}
    working.rename(columns=rename_features, inplace=True)
    final_columns = list(keep_meta) + list(rename_features.values())
    return working[final_columns].copy()


def load_player_games(raw_dir: Path, seasons: Sequence[str], available_games: pd.DataFrame) -> pd.DataFrame:
    game_keys = available_games[["season", "game_id"]].drop_duplicates().copy()
    frames = []
    for season in seasons:
        season_dir = raw_dir / f"season={season}"
        anchor = load_part_files(season_dir / "endpoint=advanced_v3" / "entity=player")
        if anchor.empty:
            continue
        anchor = normalize_player_endpoint(anchor, "advanced_v3")
        anchor = anchor[anchor["minutes_played"] > 0].copy()
        season_df = anchor.merge(game_keys, on=["season", "game_id"], how="inner")

        for endpoint in PLAYER_ENDPOINTS:
            if endpoint == "advanced_v3":
                continue
            endpoint_dir = season_dir / f"endpoint={endpoint}" / "entity=player"
            if not endpoint_dir.exists():
                continue
            extra = load_part_files(endpoint_dir)
            if extra.empty:
                continue
            extra = normalize_player_endpoint(extra, endpoint)
            merge_columns = [
                column
                for column in extra.columns
                if column
                not in {
                    "season",
                    "season_type",
                    "game_date",
                    "game_id",
                    "team_id",
                    "person_id",
                    "first_name",
                    "family_name",
                    "player_name",
                    "position",
                    "comment",
                    "minutes_raw",
                }
            ]
            season_df = season_df.merge(
                extra[["season", "game_id", "team_id", "person_id"] + merge_columns],
                on=["season", "game_id", "team_id", "person_id"],
                how="left",
                suffixes=("", f"_{endpoint}"),
            )

        frames.append(season_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_team_games(raw_dir: Path, seasons: Sequence[str], available_games: pd.DataFrame) -> pd.DataFrame:
    game_keys = available_games[["season", "game_id"]].drop_duplicates().copy()
    frames = []
    for season in seasons:
        season_dir = raw_dir / f"season={season}"
        anchor = load_part_files(season_dir / "endpoint=advanced_v3" / "entity=team")
        if anchor.empty:
            continue
        season_df = normalize_team_endpoint(anchor, "advanced_v3").merge(
            game_keys, on=["season", "game_id"], how="inner"
        )

        for endpoint in TEAM_ENDPOINTS:
            if endpoint == "advanced_v3":
                continue
            endpoint_dir = season_dir / f"endpoint={endpoint}" / "entity=team"
            if not endpoint_dir.exists():
                continue
            extra = load_part_files(endpoint_dir)
            if extra.empty:
                continue
            extra = normalize_team_endpoint(extra, endpoint)
            merge_columns = [
                column
                for column in extra.columns
                if column
                not in {
                    "season",
                    "season_type",
                    "game_date",
                    "game_id",
                    "team_id",
                    "minutes_raw",
                }
            ]
            season_df = season_df.merge(
                extra[["season", "game_id", "team_id"] + merge_columns],
                on=["season", "game_id", "team_id"],
                how="left",
                suffixes=("", f"_{endpoint}"),
            )

        frames.append(season_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def select_source_feature_columns(df: pd.DataFrame, include_columns: Sequence[str] = ()) -> list[str]:
    columns = []
    for column in df.columns:
        if column in include_columns:
            columns.append(column)
            continue
        if "__" not in column:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def build_team_schedule(available_games: pd.DataFrame) -> pd.DataFrame:
    catalog = available_games.copy()
    catalog["game_date"] = pd.to_datetime(catalog["game_date"], utc=False)
    catalog["home_team_id"] = pd.to_numeric(catalog["home_team_id"], errors="coerce").astype("Int64")
    catalog["visitor_team_id"] = pd.to_numeric(catalog["visitor_team_id"], errors="coerce").astype("Int64")
    catalog["home_points"] = pd.to_numeric(catalog["home_points"], errors="coerce")
    catalog["visitor_points"] = pd.to_numeric(catalog["visitor_points"], errors="coerce")

    base_columns = ["season", "season_type", "game_id", "game_date"]
    home = catalog[base_columns].copy()
    home["team_id"] = catalog["home_team_id"]
    home["opponent_team_id"] = catalog["visitor_team_id"]
    home["side"] = "home"
    home["is_home"] = 1
    home["team_points"] = catalog["home_points"]
    home["opponent_points"] = catalog["visitor_points"]

    away = catalog[base_columns].copy()
    away["team_id"] = catalog["visitor_team_id"]
    away["opponent_team_id"] = catalog["home_team_id"]
    away["side"] = "away"
    away["is_home"] = 0
    away["team_points"] = catalog["visitor_points"]
    away["opponent_points"] = catalog["home_points"]

    schedule = pd.concat([home, away], ignore_index=True)
    schedule["point_diff"] = schedule["team_points"] - schedule["opponent_points"]
    schedule["team_win"] = (schedule["point_diff"] > 0).astype(int)
    schedule.sort_values(["season", "team_id", "game_date", "game_id"], inplace=True)
    schedule["team_game_number"] = schedule.groupby(["season", "team_id"]).cumcount() + 1
    schedule["team_games_played_before"] = schedule["team_game_number"] - 1
    schedule["team_days_rest"] = (
        schedule.groupby(["season", "team_id"])["game_date"].diff().dt.total_seconds() / 86400.0
    )
    return schedule.reset_index(drop=True)


def build_team_features(team_schedule: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    stats = team_stats.copy()
    stats["game_date"] = pd.to_datetime(stats["game_date"], utc=False)
    merged = team_schedule.merge(
        stats.drop(columns=["season_type", "game_date"], errors="ignore"),
        on=["season", "game_id", "team_id"],
        how="left",
    )
    merged.sort_values(["season", "team_id", "game_date", "game_id"], inplace=True)

    source_columns = select_source_feature_columns(merged, include_columns=("minutes_played",))
    source_columns += ["team_points", "opponent_points", "point_diff", "team_win"]
    source_columns = list(dict.fromkeys(source_columns))

    group_keys = ["season", "team_id"]
    shifted = merged.groupby(group_keys, sort=False)[source_columns].shift(1)
    shifted_grouped = shifted.groupby([merged["season"], merged["team_id"]], sort=False)
    season_means = shifted_grouped.expanding().mean().reset_index(level=[0, 1], drop=True)
    season_means.columns = [f"pregame_season_mean__{column}" for column in source_columns]
    last5_means = shifted_grouped.rolling(5, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    last5_means.columns = [f"pregame_last5_mean__{column}" for column in source_columns]
    merged = pd.concat([merged, season_means, last5_means], axis=1)

    keep_columns = [
        "season",
        "season_type",
        "game_id",
        "game_date",
        "team_id",
        "opponent_team_id",
        "side",
        "is_home",
        "team_game_number",
        "team_games_played_before",
        "team_days_rest",
    ] + [column for column in merged.columns if column.startswith("pregame_")]
    return merged[keep_columns].copy()


def build_player_postgame_states(player_games: pd.DataFrame, team_schedule: pd.DataFrame) -> pd.DataFrame:
    if player_games.empty:
        return pd.DataFrame()

    team_numbers = team_schedule[
        ["season", "game_id", "team_id", "team_game_number", "team_games_played_before", "game_date"]
    ].copy()
    merged = player_games.copy()
    merged["game_date"] = pd.to_datetime(merged["game_date"], utc=False)
    merged = merged.merge(
        team_numbers.drop(columns=["game_date"]),
        on=["season", "game_id", "team_id"],
        how="inner",
    )
    merged.sort_values(["season", "team_id", "person_id", "game_date", "game_id"], inplace=True)

    group_keys = ["season", "team_id", "person_id"]
    grouped = merged.groupby(group_keys, sort=False)
    merged["player_games_played"] = grouped.cumcount() + 1
    merged["last_player_game_date"] = merged["game_date"]

    source_columns = select_source_feature_columns(merged, include_columns=("minutes_played",))
    grouped_values = merged.groupby(group_keys, sort=False)[source_columns]
    season_means = grouped_values.expanding().mean().reset_index(level=[0, 1, 2], drop=True)
    season_means.columns = [f"state_season_mean__{column}" for column in source_columns]
    last5_means = grouped_values.rolling(5, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True)
    last5_means.columns = [f"state_last5_mean__{column}" for column in source_columns]
    merged = pd.concat([merged, season_means, last5_means], axis=1)

    keep_columns = [
        "season",
        "team_id",
        "person_id",
        "team_game_number",
        "player_games_played",
        "last_player_game_date",
        "player_name",
        "position",
    ] + [column for column in merged.columns if column.startswith("state_")]
    return merged[keep_columns].copy()


def build_player_candidates(team_features: pd.DataFrame, player_states: pd.DataFrame) -> pd.DataFrame:
    if player_states.empty:
        return pd.DataFrame()

    latest_names = (
        player_states.sort_values(
            ["season", "team_id", "person_id", "team_game_number", "player_games_played"],
            kind="stable",
        )
        .drop_duplicates(["season", "team_id", "person_id"], keep="last")
        [["season", "team_id", "person_id", "player_name", "position"]]
        .copy()
    )

    team_games = team_features[
        [
            "season",
            "season_type",
            "game_id",
            "game_date",
            "team_id",
            "opponent_team_id",
            "side",
            "is_home",
            "team_game_number",
            "team_games_played_before",
            "team_days_rest",
        ]
    ].copy()
    candidates = team_games.merge(latest_names, on=["season", "team_id"], how="left")
    candidates.sort_values(["season", "team_id", "person_id", "team_game_number"], inplace=True)
    candidates = candidates[candidates["person_id"].notna()].copy()

    states = player_states.copy()
    states.sort_values(["season", "team_id", "person_id", "team_game_number"], inplace=True)
    state_value_columns = [
        column
        for column in states.columns
        if column not in {"season", "team_id", "person_id", "team_game_number", "player_name", "position"}
    ]
    state_groups = {
        key: group.reset_index(drop=True)
        for key, group in states.groupby(["season", "team_id", "person_id"], sort=False)
    }

    merged_frames = []
    for key, candidate_group in candidates.groupby(["season", "team_id", "person_id"], sort=False):
        state_group = state_groups.get(key)
        candidate_group = candidate_group.sort_values("team_game_number").copy()
        if state_group is None or state_group.empty:
            empty_state = pd.DataFrame(index=candidate_group.index, columns=state_value_columns)
            merged_frames.append(pd.concat([candidate_group, empty_state], axis=1))
            continue

        candidate_numbers = candidate_group["team_game_number"].to_numpy(dtype=np.int64)
        state_numbers = state_group["team_game_number"].to_numpy(dtype=np.int64)
        state_index = np.searchsorted(state_numbers, candidate_numbers, side="left") - 1
        valid = state_index >= 0

        matched = state_group.iloc[np.clip(state_index, 0, None)][state_value_columns].reset_index(drop=True)
        matched.index = candidate_group.index
        matched.loc[~valid, :] = np.nan
        merged_frames.append(pd.concat([candidate_group, matched], axis=1))

    merged = pd.concat(merged_frames, ignore_index=True) if merged_frames else pd.DataFrame()
    merged.rename(columns={"player_games_played": "player_games_played_before"}, inplace=True)
    merged["days_since_last_player_game"] = (
        merged["game_date"] - pd.to_datetime(merged["last_player_game_date"], utc=False)
    ).dt.total_seconds() / 86400.0
    return merged


def build_player_sets(team_features: pd.DataFrame, player_states: pd.DataFrame, top_k_players: int) -> pd.DataFrame:
    base_context = team_features[
        [
            "season",
            "season_type",
            "game_id",
            "game_date",
            "team_id",
            "opponent_team_id",
            "side",
            "is_home",
            "team_game_number",
            "team_games_played_before",
            "team_days_rest",
        ]
    ].copy()

    if player_states.empty:
        all_slots = base_context.loc[base_context.index.repeat(top_k_players)].reset_index(drop=True)
        all_slots["slot_index"] = np.tile(np.arange(top_k_players), len(base_context))
        all_slots["is_padding"] = 1
        return all_slots

    candidates = build_player_candidates(team_features, player_states)
    candidates = candidates[candidates["player_games_played_before"].notna()].copy()
    candidates["selection_last5_minutes"] = candidates["state_last5_mean__minutes_played"].fillna(-1.0)
    candidates["selection_season_minutes"] = candidates["state_season_mean__minutes_played"].fillna(-1.0)
    candidates["selection_games_played"] = candidates["player_games_played_before"].fillna(0.0)

    candidates.sort_values(
        [
            "season",
            "game_id",
            "team_id",
            "selection_last5_minutes",
            "selection_season_minutes",
            "selection_games_played",
            "person_id",
        ],
        ascending=[True, True, True, False, False, False, True],
        kind="stable",
        inplace=True,
    )
    candidates["slot_index"] = candidates.groupby(["season", "game_id", "team_id"]).cumcount()
    selected = candidates[candidates["slot_index"] < top_k_players].copy()

    rename_map = {}
    for column in selected.columns:
        if column.startswith("state_"):
            rename_map[column] = f"pregame_{column[len('state_'):]}"
    selected.rename(columns=rename_map, inplace=True)

    all_slots = base_context.loc[base_context.index.repeat(top_k_players)].reset_index(drop=True)
    all_slots["slot_index"] = np.tile(np.arange(top_k_players), len(base_context))
    player_sets = all_slots.merge(
        selected.drop(columns=["team_game_number", "team_games_played_before", "team_days_rest"], errors="ignore"),
        on=[
            "season",
            "season_type",
            "game_id",
            "game_date",
            "team_id",
            "opponent_team_id",
            "side",
            "is_home",
            "slot_index",
        ],
        how="left",
    )
    player_sets["is_padding"] = player_sets["person_id"].isna().astype(int)
    return player_sets


def build_games_table(
    available_games: pd.DataFrame,
    team_features: pd.DataFrame,
    player_sets: pd.DataFrame,
    min_team_history_games: int,
    top_k_players: int,
) -> pd.DataFrame:
    games = available_games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], utc=False)
    games["home_team_id"] = pd.to_numeric(games["home_team_id"], errors="coerce").astype("Int64")
    games["visitor_team_id"] = pd.to_numeric(games["visitor_team_id"], errors="coerce").astype("Int64")
    games["home_points"] = pd.to_numeric(games["home_points"], errors="coerce")
    games["visitor_points"] = pd.to_numeric(games["visitor_points"], errors="coerce")
    games["home_win"] = (games["home_points"] > games["visitor_points"]).astype(int)

    team_key_columns = ["season", "game_id", "team_id"]
    team_meta = team_features[
        [
            "season",
            "game_id",
            "team_id",
            "team_games_played_before",
            "team_days_rest",
        ]
    ].copy()
    home_meta = team_meta.rename(
        columns={
            "team_id": "home_team_id",
            "team_games_played_before": "home_team_games_played_before",
            "team_days_rest": "home_team_days_rest",
        }
    )
    away_meta = team_meta.rename(
        columns={
            "team_id": "visitor_team_id",
            "team_games_played_before": "away_team_games_played_before",
            "team_days_rest": "away_team_days_rest",
        }
    )

    selected_counts = (
        player_sets.groupby(team_key_columns, dropna=False)["is_padding"]
        .apply(lambda series: int((series == 0).sum()))
        .reset_index(name="selected_player_count")
    )
    home_counts = selected_counts.rename(
        columns={"team_id": "home_team_id", "selected_player_count": "home_selected_player_count"}
    )
    away_counts = selected_counts.rename(
        columns={"team_id": "visitor_team_id", "selected_player_count": "away_selected_player_count"}
    )

    games = games.merge(home_meta, on=["season", "game_id", "home_team_id"], how="left")
    games = games.merge(away_meta, on=["season", "game_id", "visitor_team_id"], how="left")
    games = games.merge(home_counts, on=["season", "game_id", "home_team_id"], how="left")
    games = games.merge(away_counts, on=["season", "game_id", "visitor_team_id"], how="left")
    games["home_selected_player_count"] = games["home_selected_player_count"].fillna(0).astype(int)
    games["away_selected_player_count"] = games["away_selected_player_count"].fillna(0).astype(int)
    games["eligible_default"] = (
        (games["home_team_games_played_before"] >= min_team_history_games)
        & (games["away_team_games_played_before"] >= min_team_history_games)
        & (games["home_selected_player_count"] >= top_k_players)
        & (games["away_selected_player_count"] >= top_k_players)
    ).astype(int)
    return games


def finalize_numeric_columns(df: pd.DataFrame, keep_nan_columns: Sequence[str] = ()) -> pd.DataFrame:
    if df.empty:
        return df
    keep_nan = set(keep_nan_columns)
    result = df.copy()
    for column in result.columns:
        if column in keep_nan:
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].fillna(0.0)
    return result


def write_outputs(output_dir: Path, games: pd.DataFrame, team_features: pd.DataFrame, player_sets: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    games.to_parquet(output_dir / "games.parquet", index=False)
    team_features.to_parquet(output_dir / "team_features.parquet", index=False)
    player_sets.to_parquet(output_dir / "player_sets.parquet", index=False)


def preprocess_dataset(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    seasons = args.seasons or list_available_seasons(raw_dir)
    if not seasons:
        raise ValueError(f"No seasons found under {raw_dir}")

    logging.info("Building complete-game catalog for %s", ", ".join(seasons))
    available_games = build_available_games(raw_dir, seasons, CORE_REQUIRED_ENDPOINTS)
    if available_games.empty:
        raise ValueError("No games have all required endpoints available.")

    logging.info("Loading merged player and team game stats")
    player_games = load_player_games(raw_dir, seasons, available_games)
    team_stats = load_team_games(raw_dir, seasons, available_games)

    logging.info("Building leakage-safe pregame features")
    team_schedule = build_team_schedule(available_games)
    team_features = build_team_features(team_schedule, team_stats)
    player_states = build_player_postgame_states(player_games, team_schedule)
    player_sets = build_player_sets(team_features, player_states, args.top_k_players)
    games = build_games_table(
        available_games,
        team_features,
        player_sets,
        args.min_team_history_games,
        args.top_k_players,
    )

    logging.info("Finalizing output tables")
    team_features = finalize_numeric_columns(team_features, keep_nan_columns=("team_days_rest",))
    player_sets = finalize_numeric_columns(player_sets, keep_nan_columns=("days_since_last_player_game",))
    games = finalize_numeric_columns(
        games,
        keep_nan_columns=("home_team_days_rest", "away_team_days_rest"),
    )

    write_outputs(output_dir, games, team_features, player_sets)
    logging.info(
        "Wrote %s games, %s team rows, %s player rows to %s",
        len(games),
        len(team_features),
        len(player_sets),
        output_dir,
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    preprocess_dataset(args)


if __name__ == "__main__":
    main()
