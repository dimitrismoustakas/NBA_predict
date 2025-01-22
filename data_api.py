import pandas as pd
from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreAdvancedV2, BoxScoreFourFactorsV2, BoxScoreUsageV2, BoxScoreScoringV2
import time
from tqdm import tqdm
import logging
from requests.exceptions import Timeout, ConnectionError
import sys
import numpy as np
import os
import json
import pyarrow
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from threading import Semaphore

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("nba_data_fetcher.log")
    ]
)

# Configuration Parameters
SEASONS = ['2018-19']  # Modify seasons as needed
API_DELAY = 0.6  # Delay in seconds between API calls to respect rate limits
MAX_RETRIES = 3  # Maximum number of retries for failed requests
BACKOFF_FACTOR = 2  # Factor by which the delay increases after each retry
ADVANCED_METRICS = [
    'AST_PCT', 'AST_RATIO', 'AST_TOV', 'DEF_RATING', 'DREB_PCT',
    'EFG_PCT', 'NET_RATING', 'OFF_RATING', 'OREB_PCT', 'PACE',
    'PIE', 'REB_PCT', 'TM_TOV_PCT', 'TS_PCT', 'USG_PCT'
]
FOUR_FACTORS_METRICS = [
    'FTA_RATE', 'OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT'
]
USAGE_METRICS = [
    'PCT_FGM', 'PCT_FGA', 'PCT_FG3M', 'PCT_FG3A',
    'PCT_FTM', 'PCT_FTA', 'PCT_OREB', 'PCT_DREB',
    'PCT_REB', 'PCT_AST', 'PCT_TOV', 'PCT_STL',
    'PCT_BLK', 'PCT_BLKA', 'PCT_PF', 'PCT_PFD',
    'PCT_PTS'
]
SCORING_METRICS = [  # Added Scoring Metrics
    'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR',
    'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV',
    'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM',
    'PCT_AST_3PM', 'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM'
]
MAX_BOXSCORES = 10  # Set to the desired number of boxscores for debugging
NUM_WORKERS = 4  # Number of parallel threads (adjust based on rate limits)

class NBADataFetcher:
    def __init__(self, seasons, advanced_metrics, four_factors_metrics, usage_metrics, scoring_metrics,
                 api_delay=1, max_retries=3, backoff_factor=2, max_boxscores=None, num_workers=4):
        """
        Initialize the NBADataFetcher.

        Parameters:
        - seasons (list): List of season strings (e.g., ['2018-19']).
        - advanced_metrics (list): List of metrics to calculate medians for players.
        - four_factors_metrics (list): List of four factors metrics to calculate medians for players and teams.
        - usage_metrics (list): List of usage metrics to calculate medians for players and teams.
        - scoring_metrics (list): List of scoring metrics to calculate medians for players and teams.
        - api_delay (float): Delay between API calls in seconds.
        - max_retries (int): Maximum number of retries for failed API requests.
        - backoff_factor (int): Factor by which the delay increases after each retry.
        - max_boxscores (int, optional): Maximum number of boxscores to fetch for debugging.
        - num_workers (int): Number of parallel worker threads.
        """
        self.seasons = seasons
        self.player_advanced_metrics = advanced_metrics  # Metrics for players
        self.team_advanced_metrics = advanced_metrics + ['PTS']  # Metrics for teams, including 'PTS'
        self.four_factors_metrics = four_factors_metrics  # Additional four factors metrics
        self.usage_metrics = usage_metrics  # Additional usage metrics
        self.scoring_metrics = scoring_metrics  # Additional scoring metrics
        self.api_delay = api_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_boxscores = max_boxscores  # New parameter to limit number of boxscores
        self.num_workers = num_workers
        self.games_df = pd.DataFrame()
        self.player_stats_df = pd.DataFrame()
        self.team_stats_df = pd.DataFrame()
        self.failed_game_ids = []
        self.home_team_players = []
        self.away_team_players = []
        self.team_level_info = []
        self.semaphore = Semaphore(num_workers)  # Control the number of concurrent API calls

    def convert_min_to_float(self, min_str):
        """
        Convert 'MIN' from "MM:SS" format to total minutes as a float.
        For example, "34:12" becomes 34.2.

        Parameters:
        - min_str (str): The 'MIN' string from the API.

        Returns:
        - float: Total minutes as a float.
        """
        try:
            if pd.isna(min_str):
                return 0.0
            parts = min_str.split(':')
            if len(parts) == 2:
                return float(parts[0]) + float(parts[1])/60
            else:
                return float(min_str)
        except:
            logging.warning("Failed to convert MIN value: %s. Setting to 0.0", min_str)
            return 0.0

    def fetch_games(self):
        logging.info("Fetching games for seasons: %s", ', '.join(self.seasons))
        all_dfs = []
        total_games_fetched = 0  # Counter to track the number of games fetched

        for season in self.seasons:
            for stype in ["Regular Season", "Playoffs"]:
                logging.info("Fetching games for season: %s, type: %s", season, stype)
                retries = 0
                success = False
                while retries < self.max_retries and not success:
                    try:
                        # Query the API
                        game_finder = LeagueGameFinder(
                            player_or_team_abbreviation='T',
                            league_id_nullable='00',     # NBA
                            season_nullable=season,
                            season_type_nullable=stype
                        )
                        df = game_finder.get_data_frames()[0]
                        success = True
                        logging.info("Fetched %d games for season %s, type %s", len(df), season, stype)
                    except (Timeout, ConnectionError) as e:
                        retries += 1
                        wait_time = self.backoff_factor ** retries
                        logging.warning("Timeout/ConnectionError while fetching games for season %s, type %s: %s. Retrying in %d seconds (%d/%d)", 
                                        season, stype, e, wait_time, retries, self.max_retries)
                        time.sleep(wait_time)
                    except Exception as e:
                        logging.error("Error fetching games for season %s, type %s: %s", season, stype, e)
                        break  # Non-retriable error
                if not success:
                    logging.error("Failed to fetch games for season %s, type %s after %d retries.", season, stype, self.max_retries)
                    continue  # Skip to next season/type

                # Keep minimal columns
                required_columns = [
                    "GAME_ID", "GAME_DATE", "TEAM_ID", "PTS",
                    "MATCHUP", "SEASON_ID"
                ]
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    logging.error("Missing columns in fetched data: %s", missing_columns)
                    continue

                df = df[required_columns].copy()
                df["SEASON"] = season  # Ensure 'SEASON' matches format used elsewhere

                # Identify home vs. visitor
                df["HOME_OR_AWAY"] = df["MATCHUP"].apply(lambda x: "HOME" if "vs." in x else "AWAY")

                # Split into two DataFrames
                df_home = df[df["HOME_OR_AWAY"] == "HOME"].rename(
                    columns={
                        "TEAM_ID": "HOME_TEAM_ID",
                        "PTS": "HOME_POINTS",
                        # "WL": "HOME_RESULT"  # 'WL' might not be present; adjust accordingly
                    }
                ).drop(columns=["MATCHUP", "HOME_OR_AWAY"], errors='ignore')

                df_away = df[df["HOME_OR_AWAY"] == "AWAY"].rename(
                    columns={
                        "TEAM_ID": "VISITOR_TEAM_ID",
                        "PTS": "VISITOR_POINTS",
                        # "WL": "AWAY_RESULT"  # 'WL' might not be present; adjust accordingly
                    }
                ).drop(columns=["MATCHUP", "HOME_OR_AWAY"], errors='ignore')

                # Merge on game-level keys
                try:
                    merged = pd.merge(
                        df_home,
                        df_away,
                        on=["GAME_ID", "GAME_DATE", "SEASON_ID", "SEASON"],
                        how="inner"
                    )
                except Exception as e:
                    logging.error("Error merging home and away data for season %s, type %s: %s", season, stype, e)
                    continue

                # 'RESULT' from home team's perspective
                # if 'HOME_RESULT' in merged.columns and 'AWAY_RESULT' in merged.columns:
                #     merged["RESULT"] = merged["HOME_RESULT"]
                #     merged.drop(columns=["HOME_RESULT", "AWAY_RESULT"], inplace=True)
                # else:
                # Compute RESULT based on points
                merged["RESULT"] = np.where(merged["HOME_POINTS"] > merged["VISITOR_POINTS"], "W", "L")
                merged.drop(columns=["HOME_RESULT", "AWAY_RESULT"], inplace=True, errors='ignore')

                # Final columns
                merged = merged[[
                    "SEASON", "GAME_ID", "GAME_DATE",
                    "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                    "HOME_POINTS", "VISITOR_POINTS",
                    "RESULT"
                ]]

                all_dfs.append(merged)
                total_games_fetched += 1


        if all_dfs:
            self.games_df = pd.concat(all_dfs, ignore_index=True)
            logging.info("Total games fetched before deduplication: %d", len(self.games_df))
            self.games_df.drop_duplicates(subset='GAME_ID', inplace=True)
            logging.info("Total games after deduplication: %d", len(self.games_df))
            # Convert GAME_DATE to datetime for proper sorting
            self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
            # Sort games by SEASON and date to facilitate chronological processing
            self.games_df.sort_values(['SEASON', 'GAME_DATE'], inplace=True)
            self.games_df.reset_index(drop=True, inplace=True)
            logging.debug("games_df head:\n%s", self.games_df.head())
        else:
            logging.warning("No games fetched. Please check the season parameters or API connectivity.")

    def worker_fetch_boxscore(self, game):
        """
        Worker function to fetch boxscore data for a single game.

        Parameters:
        - game (dict): A dictionary representing a single game.

        Returns:
        - tuple: (game_id, player_stats, team_stats) or None if failed.
        """
        game_id = game['GAME_ID']
        season = game['SEASON']
        retries = 0
        success = False
        while retries < self.max_retries and not success:
            try:
                # Acquire semaphore before making API calls
                with self.semaphore:
                    # Fetch BoxScoreAdvancedV2 data
                    box_score = BoxScoreAdvancedV2(game_id=game_id, timeout=60)
                    player_stats = box_score.player_stats.get_data_frame()
                    team_stats = box_score.team_stats.get_data_frame()

                    # Fetch BoxScoreFourFactorsV2 data
                    box_score_four_factors = BoxScoreFourFactorsV2(game_id=game_id, timeout=60)
                    players_four_factors = box_score_four_factors.sql_players_four_factors.get_data_frame()
                    teams_four_factors = box_score_four_factors.sql_teams_four_factors.get_data_frame()
                    # Fetch BoxScoreUsageV2 data
                    box_score_usage = BoxScoreUsageV2(game_id=game_id, timeout=60)
                    players_usage = box_score_usage.sql_players_usage.get_data_frame()
                    teams_usage = box_score_usage.sql_teams_usage.get_data_frame()
                    # Fetch BoxScoreScoringV2 data  # Added Scoring Stats
                    box_score_scoring = BoxScoreScoringV2(game_id=game_id, timeout=60)
                    players_scoring = box_score_scoring.sql_players_scoring.get_data_frame()
                    teams_scoring = box_score_scoring.sql_teams_scoring.get_data_frame()
                # Add 'SEASON' and 'GAME_DATE' from games_df
                player_stats['SEASON'] = season
                player_stats['GAME_DATE'] = game['GAME_DATE']
                team_stats['SEASON'] = season
                team_stats['GAME_DATE'] = game['GAME_DATE']

                players_four_factors['SEASON'] = season
                players_four_factors['GAME_DATE'] = game['GAME_DATE']
                teams_four_factors['SEASON'] = season
                teams_four_factors['GAME_DATE'] = game['GAME_DATE']

                players_usage['SEASON'] = season
                players_usage['GAME_DATE'] = game['GAME_DATE']
                teams_usage['SEASON'] = season
                teams_usage['GAME_DATE'] = game['GAME_DATE']

                players_scoring['SEASON'] = season  # Added
                players_scoring['GAME_DATE'] = game['GAME_DATE']  # Added
                teams_scoring['SEASON'] = season  # Added
                teams_scoring['GAME_DATE'] = game['GAME_DATE']  # Added

                # Convert 'MIN' from "MM:SS" to float
                player_stats['MIN'] = player_stats['MIN'].apply(self.convert_min_to_float)
                players_four_factors['MIN'] = players_four_factors['MIN'].apply(self.convert_min_to_float)
                team_stats['MIN'] = team_stats['MIN'].apply(self.convert_min_to_float)
                teams_four_factors['MIN'] = teams_four_factors['MIN'].apply(self.convert_min_to_float)
                players_usage['MIN'] = players_usage['MIN'].apply(self.convert_min_to_float)
                teams_usage['MIN'] = teams_usage['MIN'].apply(self.convert_min_to_float)
                players_scoring['MIN'] = players_scoring['MIN'].apply(self.convert_min_to_float)  # Added
                teams_scoring['MIN'] = teams_scoring['MIN'].apply(self.convert_min_to_float)  # Added

                # Assign 'PTS' to team_stats by merging with games_df
                games_subset = self.games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS']]
                team_stats = team_stats.merge(games_subset, on='GAME_ID', how='left')
                team_stats['PTS'] = np.where(
                    team_stats['TEAM_ID'] == team_stats['HOME_TEAM_ID'],
                    team_stats['HOME_POINTS'],
                    team_stats['VISITOR_POINTS']
                )
                team_stats.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS'], inplace=True, errors='ignore')

                # Similarly assign 'PTS' to teams_four_factors
                teams_four_factors = teams_four_factors.merge(games_subset, on='GAME_ID', how='left')
                teams_four_factors['PTS'] = np.where(
                    teams_four_factors['TEAM_ID'] == teams_four_factors['HOME_TEAM_ID'],
                    teams_four_factors['HOME_POINTS'],
                    teams_four_factors['VISITOR_POINTS']
                )
                teams_four_factors.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS'], inplace=True, errors='ignore')

                # Similarly assign 'PTS' to teams_usage
                teams_usage = teams_usage.merge(games_subset, on='GAME_ID', how='left')
                teams_usage['PTS'] = np.where(
                    teams_usage['TEAM_ID'] == teams_usage['HOME_TEAM_ID'],
                    teams_usage['HOME_POINTS'],
                    teams_usage['VISITOR_POINTS']
                )
                teams_usage.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS'], inplace=True, errors='ignore')

                # Similarly assign 'PTS' to teams_scoring
                teams_scoring = teams_scoring.merge(games_subset, on='GAME_ID', how='left')
                teams_scoring['PTS'] = np.where(
                    teams_scoring['TEAM_ID'] == teams_scoring['HOME_TEAM_ID'],
                    teams_scoring['HOME_POINTS'],
                    teams_scoring['VISITOR_POINTS']
                )
                teams_scoring.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS'], inplace=True, errors='ignore')

                # Merge advanced stats with four factors stats for players
                player_stats = player_stats.merge(
                    players_four_factors[['GAME_ID', 'PLAYER_ID'] + self.four_factors_metrics],
                    on=['GAME_ID', 'PLAYER_ID'],
                    how='left'
                )

                # Merge advanced stats with four factors stats for teams
                team_stats = team_stats.merge(
                    teams_four_factors[['GAME_ID', 'TEAM_ID'] + self.four_factors_metrics],
                    on=['GAME_ID', 'TEAM_ID'],
                    how='left'
                )

                # Merge usage stats for players
                player_stats = player_stats.merge(
                    players_usage[['GAME_ID', 'PLAYER_ID'] + self.usage_metrics],
                    on=['GAME_ID', 'PLAYER_ID'],
                    how='left'
                )

                # Merge usage stats for teams
                team_stats = team_stats.merge(
                    teams_usage[['GAME_ID', 'TEAM_ID'] + self.usage_metrics],
                    on=['GAME_ID', 'TEAM_ID'],
                    how='left'
                )

                # Merge scoring stats for players  # Added
                player_stats = player_stats.merge(
                    players_scoring[['GAME_ID', 'PLAYER_ID'] + self.scoring_metrics],
                    on=['GAME_ID', 'PLAYER_ID'],
                    how='left'
                )

                # Merge scoring stats for teams  # Added
                team_stats = team_stats.merge(
                    teams_scoring[['GAME_ID', 'TEAM_ID'] + self.scoring_metrics],
                    on=['GAME_ID', 'TEAM_ID'],
                    how='left'
                )

                # Handle missing four factors, usage, and scoring metrics by filling with 0
                player_stats[self.four_factors_metrics + self.usage_metrics + self.scoring_metrics] = player_stats[self.four_factors_metrics + self.usage_metrics + self.scoring_metrics].fillna(0)
                team_stats[self.four_factors_metrics + self.usage_metrics + self.scoring_metrics] = team_stats[self.four_factors_metrics + self.usage_metrics + self.scoring_metrics].fillna(0)

                # Verify 'SEASON' and 'GAME_DATE' columns
                if 'SEASON' not in player_stats.columns or 'GAME_DATE' not in player_stats.columns:
                    logging.error("'SEASON' or 'GAME_DATE' missing in player_stats for GameID %s", game_id)
                    raise KeyError("'SEASON' or 'GAME_DATE' missing in player_stats")

                if 'SEASON' not in team_stats.columns or 'GAME_DATE' not in team_stats.columns or 'PTS' not in team_stats.columns:
                    logging.error("'SEASON', 'GAME_DATE', or 'PTS' missing in team_stats for GameID %s", game_id)
                    raise KeyError("'SEASON', 'GAME_DATE', or 'PTS' missing in team_stats")

                # Prepare player and team stats to return
                return (game_id, player_stats, team_stats)
            except (Timeout, ConnectionError) as e:
                retries += 1
                wait_time = self.backoff_factor ** retries
                logging.warning("Timeout/ConnectionError for GameID %s: %s. Retrying in %d seconds (%d/%d)", 
                                game_id, e, wait_time, retries, self.max_retries)
                time.sleep(wait_time)
            except KeyError as e:
                logging.error("KeyError for GameID %s: %s", game_id, e)
                self.failed_game_ids.append(game_id)
                return None  # Non-retriable error
            except Exception as e:
                logging.error("Error fetching stats for GameID %s: %s", game_id, e)
                self.failed_game_ids.append(game_id)
                return None  # Non-retriable error

    def fetch_advanced_stats(self):
        logging.info("Fetching advanced stats for boxscores using threading.")
        player_stats_list = []
        team_stats_list = []
        boxscores_fetched = 0  # Counter to track number of boxscores fetched

        # Limit the number of boxscores if max_boxscores is set
        if self.max_boxscores is not None:
            games_to_fetch = self.games_df.head(self.max_boxscores)
        else:
            games_to_fetch = self.games_df

        # Prepare list of games as dictionaries
        games = games_to_fetch.to_dict('records')

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_game = {executor.submit(self.worker_fetch_boxscore, game): game for game in games}

            with tqdm(as_completed(future_to_game), total=len(future_to_game), desc="Fetching boxscores") as pbar:
                for step, future in enumerate(pbar, 1):  # Start counting from 1
                    game = future_to_game[future]
                    try:
                        result = future.result()
                        if result:
                            game_id, player_stats, team_stats = result
                            player_stats_list.append(player_stats)
                            team_stats_list.append(team_stats)
                            boxscores_fetched += 1
                    except Exception as e:
                        logging.error("Exception occurred while processing GameID %s: %s", game['GAME_ID'], e)
                        self.failed_game_ids.append(game['GAME_ID'])
                    
                    # Check if the maximum number of boxscores has been fetched
                    if self.max_boxscores is not None and boxscores_fetched >= self.max_boxscores:
                        logging.info("Reached the maximum number of boxscores to fetch: %d", self.max_boxscores)
                        break

        if player_stats_list:
            self.player_stats_df = pd.concat(player_stats_list, ignore_index=True)
            logging.info("Player stats fetched: %d records", len(self.player_stats_df))
            logging.debug("player_stats_df columns: %s", self.player_stats_df.columns.tolist())
        else:
            logging.warning("No player stats fetched.")
        if team_stats_list:
            self.team_stats_df = pd.concat(team_stats_list, ignore_index=True)
            logging.info("Team stats fetched: %d records", len(self.team_stats_df))
            logging.debug("team_stats_df columns: %s", self.team_stats_df.columns.tolist())
        else:
            logging.warning("No team stats fetched.")

    def calculate_medians(self):
        logging.info("Calculating seasonal medians for players and teams.")
        if self.player_stats_df.empty or self.team_stats_df.empty:
            logging.warning("Player or Team stats DataFrame is empty. Skipping median calculations.")
            return

        # Ensure 'SEASON' and 'GAME_DATE' are in player_stats_df
        required_columns = ['SEASON', 'PLAYER_ID', 'GAME_DATE']
        for col in required_columns:
            if col not in self.player_stats_df.columns:
                logging.error("Required column '%s' is missing in player_stats_df.", col)
                return  # Exit the method as further processing depends on these columns

        # Drop records with missing 'SEASON' or 'GAME_DATE'
        missing_season = self.player_stats_df['SEASON'].isnull().sum()
        missing_game_date = self.player_stats_df['GAME_DATE'].isnull().sum()
        if missing_season > 0 or missing_game_date > 0:
            logging.warning("Dropping %d records with missing 'SEASON' and %d records with missing 'GAME_DATE'.", 
                            missing_season, missing_game_date)
            self.player_stats_df.dropna(subset=['SEASON', 'GAME_DATE'], inplace=True)

        # Sort by SEASON, PLAYER_ID, and GAME_DATE to ensure chronological order within each season
        self.player_stats_df.sort_values(['SEASON', 'PLAYER_ID', 'GAME_DATE'], inplace=True)
        self.team_stats_df.sort_values(['SEASON', 'TEAM_ID', 'GAME_DATE'], inplace=True)

        # Initialize median columns for player stats
        all_player_metrics = self.player_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics  # Added scoring_metrics
        for metric in all_player_metrics:
            median_col = f'MEDIAN_PLAYER_{metric}'
            self.player_stats_df[median_col] = self.player_stats_df.groupby(['SEASON', 'PLAYER_ID'])[metric].expanding().median().reset_index(level=[0,1], drop=True)

        # Shift medians to exclude current game
        player_median_cols = [f'MEDIAN_PLAYER_{m}' for m in all_player_metrics]
        self.player_stats_df[player_median_cols] = self.player_stats_df.groupby(['SEASON', 'PLAYER_ID'])[player_median_cols].shift(1)

        # Initialize median columns for team stats
        all_team_metrics = self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics  # Added scoring_metrics
        for metric in all_team_metrics:
            median_col = f'MEDIAN_TEAM_{metric}'
            self.team_stats_df[median_col] = self.team_stats_df.groupby(['SEASON', 'TEAM_ID'])[metric].expanding().median().reset_index(level=[0,1], drop=True)

        # Shift medians to exclude current game
        team_median_cols = [f'MEDIAN_TEAM_{m}' for m in all_team_metrics]
        self.team_stats_df[team_median_cols] = self.team_stats_df.groupby(['SEASON', 'TEAM_ID'])[team_median_cols].shift(1)

        if 'MIN' in self.player_stats_df.columns:
            logging.info("Calculating average minutes for players.")
            self.player_stats_df['Average_Minutes'] = (
                self.player_stats_df.groupby(['SEASON', 'PLAYER_ID'])['MIN']
                .expanding()
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            # Shift 'Average_Minutes' to exclude the current game
            self.player_stats_df['Average_Minutes'] = self.player_stats_df.groupby(['SEASON', 'PLAYER_ID'])['Average_Minutes'].shift(1)
        else:
            logging.warning("Column 'MIN' is missing in player_stats_df. Skipping average minutes calculation.")

        logging.debug("player_stats_df head after calculating medians:\n%s", self.player_stats_df.head())
        logging.debug("team_stats_df head after calculating medians:\n%s", self.team_stats_df.head())

    def select_top_players(self):
        logging.info("Selecting top 10 players per team based on median MIN.")
        if self.player_stats_df.empty:
            logging.warning("Player stats DataFrame is empty. Skipping player selection.")
            return

        # Merge player_stats_df with games_df to get HOME_TEAM_ID and VISITOR_TEAM_ID
        merged_player = self.player_stats_df.merge(
            self.games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']],
            on='GAME_ID',
            how='left'
        )

        logging.debug("merged_player columns after merge: %s", merged_player.columns.tolist())

        # Determine team side (Home or Away) for each player
        merged_player['TEAM_SIDE'] = np.where(
            merged_player['TEAM_ID'] == merged_player['HOME_TEAM_ID'], 'Home', 'Visitor'
        )

        # Exclude players who did not play in the current game (MIN > 0)
        merged_player_played = merged_player[merged_player['MIN'] > 0]

        # Check if 'SEASON' column exists in merged_player_played
        if 'SEASON' not in merged_player_played.columns:
            logging.error("'SEASON' column is missing in merged_player_played.")
            return

        # For each game and team side, select top 10 players based on Average_Minutes
        for game_id in tqdm(self.games_df['GAME_ID'], desc="Processing games for top players"):
            game = self.games_df[self.games_df['GAME_ID'] == game_id].iloc[0]
            season = game['SEASON']
            for side in ['Home', 'Visitor']:
                team_id = game[f'{side.upper()}_TEAM_ID']
                team_players = merged_player_played[
                    (merged_player_played['GAME_ID'] == game_id) &
                    (merged_player_played['TEAM_ID'] == team_id) &
                    (merged_player_played['SEASON'] == season)
                ]

                # Check if 'Average_Minutes' exists
                if 'Average_Minutes' not in team_players.columns:
                    logging.error("'Average_Minutes' column is missing in team_players for GameID %s, TeamID %s", game_id, team_id)
                    continue

                # Select top 10 players based on Average_Minutes
                top_players = team_players.sort_values(by='Average_Minutes', ascending=False).head(10)

                # If less than 10 players, handle accordingly (e.g., select all available)
                if top_players.shape[0] < 10:
                    n_missing = 10 - top_players.shape[0]

                    # Create a DataFrame with n_missing rows
                    padding_data = {
                        'GAME_ID': [game_id] * n_missing,
                        'TEAM_ID': [team_id] * n_missing,
                        'PLAYER_ID': ['Missing'] * n_missing,
                        'PLAYER_NAME': ['Missing'] * n_missing,
                        'Average_Minutes': [0] * n_missing
                    }

                    # Add median metrics with 0
                    for metric in self.player_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics:  # Added scoring_metrics
                        padding_data[f'MEDIAN_PLAYER_{metric}'] = [0] * n_missing

                    padding_df = pd.DataFrame(padding_data)

                    # Combine the top_players with padding_df
                    top_players = pd.concat([top_players, padding_df], ignore_index=True)

                # Ensure no more than 10 players
                top_players = top_players.head(10)

                # Select relevant columns (including GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME, and median metrics)
                selected_columns = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME', 'Average_Minutes'] + [f'MEDIAN_PLAYER_{m}' for m in self.player_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics]  # Added scoring_metrics
                # Check if all selected_columns exist
                missing_selected_columns = [col for col in selected_columns if col not in top_players.columns]
                if missing_selected_columns:
                    logging.error("Missing columns %s in top_players for GameID %s, TeamID %s", missing_selected_columns, game_id, team_id)
                    continue

                top_players_selected = top_players[selected_columns].copy()

                top_players_selected['TEAM_SIDE'] = side

                if side == 'Home':
                    self.home_team_players.append(top_players_selected)
                else:
                    self.away_team_players.append(top_players_selected)

    def prepare_team_level_info(self):
        logging.info("Preparing team-level information table.")
        if self.team_stats_df.empty:
            logging.warning("Team stats DataFrame is empty. Skipping team-level info preparation.")
            return

        # Merge team_stats_df with games_df to associate each team's stats with the corresponding game
        merged_team = self.team_stats_df.merge(
            self.games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE', 'HOME_POINTS', 'VISITOR_POINTS', 'SEASON', 'RESULT']],
            on='GAME_ID',
            how='left'
        )

        # Prepare home team median metrics
        home_team_stats = merged_team[merged_team['TEAM_ID'] == merged_team['HOME_TEAM_ID']].copy()
        home_medians = home_team_stats[['GAME_ID'] + [f'MEDIAN_TEAM_{m}' for m in self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics]].copy()
        home_medians = home_medians.rename(columns={f'MEDIAN_TEAM_{m}': f'Home_MEDIAN_{m}' for m in self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics})
        # Rename 'MEDIAN_TEAM_PTS' to 'Home_MEDIAN_PTS' if exists
        if 'MEDIAN_TEAM_PTS' in home_medians.columns:
            home_medians = home_medians.rename(columns={'MEDIAN_TEAM_PTS': 'Home_MEDIAN_PTS'})

        # Prepare visitor team median metrics
        visitor_team_stats = merged_team[merged_team['TEAM_ID'] == merged_team['VISITOR_TEAM_ID']].copy()
        visitor_medians = visitor_team_stats[['GAME_ID'] + [f'MEDIAN_TEAM_{m}' for m in self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics]].copy()
        visitor_medians = visitor_medians.rename(columns={f'MEDIAN_TEAM_{m}': f'Away_MEDIAN_{m}' for m in self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics})
        # Rename 'MEDIAN_TEAM_PTS' to 'Away_MEDIAN_PTS' if exists
        if 'MEDIAN_TEAM_PTS' in visitor_medians.columns:
            visitor_medians = visitor_medians.rename(columns={'MEDIAN_TEAM_PTS': 'Away_MEDIAN_PTS'})

        # Merge home and visitor medians on GAME_ID
        team_level = pd.merge(
            home_medians,
            visitor_medians,
            on='GAME_ID',
            how='inner'
        )

        # Merge with games_df to get additional game information
        team_level = team_level.merge(
            self.games_df[['GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS', 'SEASON', 'RESULT']],
            on='GAME_ID',
            how='left'
        )

        # Reorder columns
        median_columns = [f'Home_MEDIAN_{m}' for m in self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics] + \
                         [f'Away_MEDIAN_{m}' for m in self.team_advanced_metrics + self.four_factors_metrics + self.usage_metrics + self.scoring_metrics]
        team_level = team_level[['GAME_ID', 'SEASON', 'GAME_DATE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_POINTS', 'VISITOR_POINTS', 'RESULT'] + median_columns]

        team_level.sort_values(['SEASON', 'GAME_DATE'], inplace=True)
        team_level.reset_index(drop=True, inplace=True)

        # Append to team_level_info
        self.team_level_info = team_level.to_dict('records')

    def save_tables(self, output_dir='output'):
        logging.info("Saving the final datasets.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the Generic Games Table
        if not self.games_df.empty:
            games_filepath = os.path.join(output_dir, 'games.csv')
            self.games_df.to_csv(games_filepath, index=False)
            logging.info("Games data saved: %d records to %s", len(self.games_df), games_filepath)
        else:
            logging.warning("No games data to save.")

        # Save Player-Level Data in Nested JSON
        if not self.games_df.empty and (self.home_team_players or self.away_team_players):
            player_data = {}

            # Convert player DataFrames to dictionaries grouped by GAME_ID
            home_df = pd.concat(self.home_team_players, ignore_index=True) if self.home_team_players else pd.DataFrame()
            away_df = pd.concat(self.away_team_players, ignore_index=True) if self.away_team_players else pd.DataFrame()

            # Ensure GAME_ID is unique in games_df
            for _, game in self.games_df.iterrows():
                game_id = game['GAME_ID']
                game_info = {
                    'GAME_ID': game_id,
                    'GAME_DATE': game['GAME_DATE'].strftime('%Y-%m-%d'),
                    'SEASON': game['SEASON'],
                    'HOME_TEAM_ID': game['HOME_TEAM_ID'],
                    'VISITOR_TEAM_ID': game['VISITOR_TEAM_ID'],
                    'HOME_POINTS': game['HOME_POINTS'],
                    'VISITOR_POINTS': game['VISITOR_POINTS'],
                    'RESULT': game['RESULT'],
                    'Home_Players': [],
                    'Visitor_Players': []
                }

                # Get home players for this game
                home_players = home_df[home_df['GAME_ID'] == game_id].drop(columns=['TEAM_SIDE']) if not home_df.empty else pd.DataFrame()
                if not home_players.empty:
                    game_info['Home_Players'] = home_players.to_dict(orient='records')

                # Get visitor players for this game
                visitor_players = away_df[away_df['GAME_ID'] == game_id].drop(columns=['TEAM_SIDE']) if not away_df.empty else pd.DataFrame()
                if not visitor_players.empty:
                    game_info['Visitor_Players'] = visitor_players.to_dict(orient='records')

                player_data[game_id] = game_info

            # Save the nested player data to JSON
            player_json_path = os.path.join(output_dir, 'player_data.json')
            with open(player_json_path, 'w') as json_file:
                json.dump(player_data, json_file, indent=4, default=str)  # default=str to handle datetime serialization
            logging.info("Player-level data saved to %s", player_json_path)
        else:
            logging.warning("No player data to save.")

        # Save Team-Level Information
        if self.team_level_info:
            team_info_df = pd.DataFrame(self.team_level_info)
            team_info_filepath = os.path.join(output_dir, 'team_level_info.parquet')
            team_info_df.to_parquet(team_info_filepath, index=False)
            logging.info("Team-level information data saved: %d records to %s", len(team_info_df), team_info_filepath)
        else:
            logging.warning("No team-level information data to save.")

        # Save list of failed GameIDs
        if self.failed_game_ids:
            failed_filepath = os.path.join(output_dir, 'failed_game_ids.txt')
            with open(failed_filepath, 'w') as f:
                for game_id in self.failed_game_ids:
                    f.write(f"{game_id}\n")
            logging.info("List of failed GameIDs saved to %s", failed_filepath)
        else:
            logging.info("No failed GameIDs to save.")

    def run_pipeline(self):
        self.fetch_games()
        self.fetch_advanced_stats()
        self.calculate_medians()
        self.select_top_players()
        self.prepare_team_level_info()
        self.save_tables()

if __name__ == "__main__":
    # Initialize the fetcher with the new four factors and usage metrics
    fetcher = NBADataFetcher(
        seasons = [
    '2008-09', '2009-10', '2010-11', '2011-12', '2012-13',
    '2013-14', '2014-15', '2015-16', '2016-17', '2017-18',
    '2018-19', '2019-20', '2020-21', '2021-22',
    '2022-23', '2023-24', '2024-25'],
        advanced_metrics=ADVANCED_METRICS,
        four_factors_metrics=FOUR_FACTORS_METRICS,
        usage_metrics=USAGE_METRICS,
        scoring_metrics=SCORING_METRICS,
        api_delay=1,
        max_retries=5,
        backoff_factor=BACKOFF_FACTOR,
        num_workers = 5,
        max_boxscores=None
    )
    fetcher.run_pipeline()