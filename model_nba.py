import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from typing import List, Dict, Any
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
from collections import defaultdict
import math

# ----------------------------
# Step 1: Load the JSON Data
# ----------------------------

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Loaded JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# ----------------------------
# Step 2: Exclude First 10 Games per Team per Season
# ----------------------------

def get_excluded_game_ids(data: Dict[str, Any]) -> set:
    """
    Identify game IDs that are among the first 10 games for any team in each season.

    Args:
        data (Dict[str, Any]): Raw JSON data.

    Returns:
        set: Set of game IDs to exclude.
    """
    # Structure: {season: {team_id: list of (game_date, game_id)}}
    season_team_games = defaultdict(lambda: defaultdict(list))

    for game_id, game in data.items():
        season = game.get("SEASON")
        game_date_str = game.get("GAME_DATE")
        try:
            game_date = datetime.strptime(game_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid date format for game {game_id}: {game_date_str}. Skipping.")
            continue

        home_team_id = game.get("HOME_TEAM_ID")
        visitor_team_id = game.get("VISITOR_TEAM_ID")

        # Append game to home team
        if home_team_id is not None:
            season_team_games[season][home_team_id].append((game_date, game_id))
        # Append game to visitor team
        if visitor_team_id is not None:
            season_team_games[season][visitor_team_id].append((game_date, game_id))

    excluded_game_ids = set()

    for season, teams in season_team_games.items():
        for team_id, games in teams.items():
            # Sort games by date
            sorted_games = sorted(games, key=lambda x: x[0])
            # Get first 10 game IDs
            first_10_games = sorted_games[:10]
            for _, game_id in first_10_games:
                excluded_game_ids.add(game_id)

    print(f"Total games to exclude (first 10 per team per season): {len(excluded_game_ids)}")
    return excluded_game_ids

# ----------------------------
# Step 3: Preprocess the Data
# ----------------------------

# Define the player features you want to use
PLAYER_FEATURES = [
    'Average_Minutes', 'MEDIAN_PLAYER_AST_PCT', 'MEDIAN_PLAYER_AST_RATIO', 'MEDIAN_PLAYER_AST_TOV',
    'MEDIAN_PLAYER_DEF_RATING', 'MEDIAN_PLAYER_DREB_PCT', 'MEDIAN_PLAYER_EFG_PCT',
    'MEDIAN_PLAYER_NET_RATING', 'MEDIAN_PLAYER_OFF_RATING', 'MEDIAN_PLAYER_OREB_PCT',
    'MEDIAN_PLAYER_PACE', 'MEDIAN_PLAYER_PIE', 'MEDIAN_PLAYER_REB_PCT', 'MEDIAN_PLAYER_TM_TOV_PCT',
    'MEDIAN_PLAYER_TS_PCT', 'MEDIAN_PLAYER_USG_PCT', 'MEDIAN_PLAYER_FTA_RATE', 'MEDIAN_PLAYER_OPP_EFG_PCT',
    'MEDIAN_PLAYER_OPP_FTA_RATE', 'MEDIAN_PLAYER_OPP_TOV_PCT', 'MEDIAN_PLAYER_OPP_OREB_PCT',
    'MEDIAN_PLAYER_PCT_FGM', 'MEDIAN_PLAYER_PCT_FGA', 'MEDIAN_PLAYER_PCT_FG3M', 'MEDIAN_PLAYER_PCT_FG3A',
    'MEDIAN_PLAYER_PCT_FTM', 'MEDIAN_PLAYER_PCT_FTA', 'MEDIAN_PLAYER_PCT_OREB', 'MEDIAN_PLAYER_PCT_DREB',
    'MEDIAN_PLAYER_PCT_REB', 'MEDIAN_PLAYER_PCT_AST', 'MEDIAN_PLAYER_PCT_TOV', 'MEDIAN_PLAYER_PCT_STL',
    'MEDIAN_PLAYER_PCT_BLK', 'MEDIAN_PLAYER_PCT_BLKA', 'MEDIAN_PLAYER_PCT_PF', 'MEDIAN_PLAYER_PCT_PFD',
    'MEDIAN_PLAYER_PCT_PTS', 'MEDIAN_PLAYER_PCT_FGA_2PT', 'MEDIAN_PLAYER_PCT_FGA_3PT', 'MEDIAN_PLAYER_PCT_PTS_2PT', 
    'MEDIAN_PLAYER_PCT_PTS_2PT_MR', 'MEDIAN_PLAYER_PCT_PTS_3PT', 'MEDIAN_PLAYER_PCT_PTS_FB', 'MEDIAN_PLAYER_PCT_PTS_FT', 
    'MEDIAN_PLAYER_PCT_PTS_OFF_TOV', 'MEDIAN_PLAYER_PCT_PTS_PAINT', 'MEDIAN_PLAYER_PCT_AST_2PM', 'MEDIAN_PLAYER_PCT_UAST_2PM', 
    'MEDIAN_PLAYER_PCT_AST_3PM', 'MEDIAN_PLAYER_PCT_UAST_3PM', 'MEDIAN_PLAYER_PCT_AST_FGM', 'MEDIAN_PLAYER_PCT_UAST_FGM'
]

def preprocess_data(data: Dict[str, Any], excluded_game_ids: set) -> List[Dict]:
    """
    Preprocess the raw JSON data into a list of game dictionaries with
    home and visitor team features and labels, excluding specified games.

    This function also removes any games that contain NaN values in player features.

    Args:
        data (Dict[str, Any]): Raw JSON data.
        excluded_game_ids (set): Set of game IDs to exclude.

    Returns:
        List[Dict]: List of processed game data without excluded games and NaN values.
    """
    processed = []
    skipped_games = 0
    skipped_nan_games = 0
    excluded_games = 0

    for game_id, game in data.items():
        if game_id in excluded_game_ids:
            excluded_games += 1
            continue  # Skip excluded games

        home_players = game.get("Home_Players", [])
        visitor_players = game.get("Visitor_Players", [])

        if len(home_players) != 10 or len(visitor_players) != 10:
            print(f"Skipping game {game_id} due to incorrect number of players.")
            skipped_games += 1
            continue  # Skip games that do not have exactly 10 players per team

        # Extract player features
        home_features = []
        for player in home_players:
            features = []
            for feature in PLAYER_FEATURES:
                value = player.get(feature, None)
                if value is None:
                    features.append(np.nan)
                else:
                    features.append(value)
            home_features.append(features)

        visitor_features = []
        for player in visitor_players:
            features = []
            for feature in PLAYER_FEATURES:
                value = player.get(feature, None)
                if value is None:
                    features.append(np.nan)
                else:
                    features.append(value)
            visitor_features.append(features)

        # Convert to NumPy arrays for NaN checking
        home_array = np.array(home_features, dtype=np.float32)
        visitor_array = np.array(visitor_features, dtype=np.float32)

        # Check for NaN in home or visitor features
        if np.isnan(home_array).any() or np.isnan(visitor_array).any():
            print(f"Skipping game {game_id} due to NaN values in player features.")
            skipped_nan_games += 1
            continue  # Skip games with any NaN in player features

        # Determine the label
        result = game.get("RESULT", None)
        if result == "W":
            label = 1
        elif result == "L":
            label = 0
        else:
            print(f"Unknown result '{result}' for game {game_id}. Skipping.")
            skipped_games += 1
            continue 

        processed.append({
            "home": home_features,        # [10, num_features]
            "visitor": visitor_features,  # [10, num_features]
            "label": label
        })

    print(f"\nTotal games processed: {len(processed)}")
    print(f"Total games excluded (first 10 per team per season): {excluded_games}")
    print(f"Total games skipped due to incorrect player counts or unknown results: {skipped_games}")
    print(f"Total games skipped due to NaN values: {skipped_nan_games}\n")
    return processed

# ----------------------------
# Step 4: Define PyTorch Dataset
# ----------------------------

class GamesDataset(Dataset):
    """
    PyTorch Dataset for NBA games.
    """
    def __init__(self, data: List[Dict], player_features: List[str]):
        """
        Initialize the dataset.

        Args:
            data (List[Dict]): List of processed game data.
            player_features (List[str]): List of player feature names.
        """
        self.data = data
        self.player_features = player_features
        self.num_features = len(player_features)
        self.num_players = 10  # Fixed number of players per team

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the data for a single game.

        Args:
            idx (int): Index of the game.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (home_team_features, visitor_team_features, label)
        """
        game = self.data[idx]
        home = np.array(game['home'], dtype=np.float32)     # [10, num_features]
        visitor = np.array(game['visitor'], dtype=np.float32)  # [10, num_features]
        label = np.array(game['label'], dtype=np.float32)  # Shape: []

        # Convert to tensors
        home_tensor = torch.tensor(home, dtype=torch.float32)         # [10, num_features]
        visitor_tensor = torch.tensor(visitor, dtype=torch.float32)   # [10, num_features]
        label_tensor = torch.tensor(label, dtype=torch.float32)       # Scalar

        return home_tensor, visitor_tensor, label_tensor

# ----------------------------
# Step 5: Define the Neural Network
# ----------------------------

class DualHeadNetwork(nn.Module):
    """
    Neural network with two heads for home and visitor teams, followed by additional layers
    for final prediction.
    """
    def __init__(self, num_features: int, player_embedding_dim: int = 64, 
                 team_embedding_dim: int = 128, hidden_dim: int = 64):
        """
        Initialize the network.

        Args:
            num_features (int): Number of features per player.
            player_embedding_dim (int, optional): Dimension of player embeddings. Defaults to 64.
            team_embedding_dim (int, optional): Dimension of team embeddings. Defaults to 128.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 64.
        """
        super(DualHeadNetwork, self).__init__()

        self.num_players = 10  # Fixed number of players per team

        # Per-player processing (shared for both teams)
        self.player_mlp = nn.Sequential(
            nn.Linear(num_features, player_embedding_dim),
            nn.ReLU(),
            nn.Linear(player_embedding_dim, player_embedding_dim),
            nn.ReLU()
        )

        # Home team head
        self.home_head = nn.Sequential(
            nn.Linear(player_embedding_dim * self.num_players, team_embedding_dim),
            nn.ReLU(),
            nn.Linear(team_embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # Visitor team head
        self.visitor_head = nn.Sequential(
            nn.Linear(player_embedding_dim * self.num_players, team_embedding_dim),
            nn.ReLU(),
            nn.Linear(team_embedding_dim, hidden_dim),
            nn.ReLU()
        )

        # Final layers with additional transformations
        self.final = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),   # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 16),                # 64 -> 16
            nn.ReLU(),
            nn.Linear(16, 1)                  # 16 -> 1
            # nn.Sigmoid()  # Removed to use BCEWithLogitsLoss
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Custom weight initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights using Kaiming Uniform (default)
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)

        # Set the bias of the final linear layer to log(0.6 / 0.4) ≈ 0.405
        final_linear = self.final[-1]  # Access the last layer in the final Sequential
        with torch.no_grad():
            final_linear.bias.fill_(math.log(0.6 / 0.4))  # ≈ 0.405

            # Optional: Initialize the weights of the final layer to zero
            # This ensures that the output is solely determined by the bias initially
            final_linear.weight.fill_(0.0)

    def forward(self, home, visitor):
        """
        Forward pass of the network.

        Args:
            home (Tensor): Home team features [batch_size, 10, num_features].
            visitor (Tensor): Visitor team features [batch_size, 10, num_features].

        Returns:
            Tensor: Prediction logits [batch_size].
        """
        # Process home players
        home_embedded = self.player_mlp(home)  # [batch_size, 10, player_embedding_dim]
        home_flat = home_embedded.view(home_embedded.size(0), -1)  # [batch_size, 10 * player_embedding_dim]
        home_rep = self.home_head(home_flat)  # [batch_size, hidden_dim]

        # Process visitor players
        visitor_embedded = self.player_mlp(visitor)  # [batch_size, 10, player_embedding_dim]
        visitor_flat = visitor_embedded.view(visitor_embedded.size(0), -1)  # [batch_size, 10 * player_embedding_dim]
        visitor_rep = self.visitor_head(visitor_flat)  # [batch_size, hidden_dim]

        # Combine representations
        combined = torch.cat([home_rep, visitor_rep], dim=1)  # [batch_size, hidden_dim * 2]

        # Pass through the final layers
        out = self.final(combined)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]

# ----------------------------
# Step 6: Training Loop with Optimizer and Scheduler
# ----------------------------

def validate_labels(data: List[Dict]):
    """
    Validate that all labels are either 0 or 1.

    Args:
        data (List[Dict]): List of processed game data.

    Raises:
        ValueError: If any label is not 0 or 1.
    """
    labels = [game['label'] for game in data]
    unique_labels = set(labels)
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid labels found: {unique_labels}")

def train_model(model, dataloader, val_loader, epochs=10, lr=1e-3, device='cpu',
               weight_decay=1e-2, betas=(0.9, 0.999), max_grad_norm=1.0,
               warmup_iters_ratio=0.1, min_lr=1e-6):
    """
    Train the neural network with AdamW optimizer, Cosine Decay with Warmup,
    weight decay, betas, and gradient clipping. Also records and plots the learning rate.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        lr (float, optional): Initial learning rate. Defaults to 1e-3.
        device (str, optional): Device to train on ('cpu' or 'cuda'). Defaults to 'cpu'.
        weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-2.
        betas (tuple, optional): Betas for AdamW optimizer. Defaults to (0.9, 0.999).
        max_grad_norm (float, optional): Maximum norm for gradient clipping. Defaults to 1.0.
        warmup_iters_ratio (float, optional): Ratio of total iterations for warmup. Defaults to 0.1.
        min_lr (float, optional): Minimum learning rate after decay. Defaults to 1e-6.
    """
    # Initialize optimizer with AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    # Calculate total iterations and warmup iterations
    total_iterations = epochs * len(dataloader)
    warmup_iters = int(warmup_iters_ratio * total_iterations)

    # Define a list to store learning rates
    learning_rates = []

    # Initialize iteration counter
    iter_num = 0

    # Use BCEWithLogitsLoss instead of BCELoss
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []

        for home, visitor, labels in dataloader:
            iter_num += 1  # Increment iteration counter

            home = home.to(device)           # [batch_size, 10, num_features]
            visitor = visitor.to(device)     # [batch_size, 10, num_features]
            labels = labels.to(device)       # [batch_size]

            optimizer.zero_grad()
            outputs = model(home, visitor)   # [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update learning rate using Cosine Decay with Warmup
            if iter_num < warmup_iters:
                lr_current = lr * (iter_num + 1) / (warmup_iters + 1)
            elif iter_num > total_iterations:
                lr_current = min_lr
            else:
                decay_ratio = (iter_num - warmup_iters) / (total_iterations - warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                lr_current = min_lr + coeff * (lr - min_lr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_current

            # Record the current learning rate
            learning_rates.append(lr_current)

            optimizer.step()

            epoch_loss += loss.item() * home.size(0)
            # Apply sigmoid to outputs for metric calculations
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = epoch_loss / len(dataloader.dataset)
        preds_binary = [1 if p >= 0.5 else 0 for p in all_preds]
        accuracy = accuracy_score(all_labels, preds_binary)
        roc_auc = roc_auc_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for home, visitor, labels in val_loader:
                home = home.to(device)
                visitor = visitor.to(device)
                labels = labels.to(device)

                outputs = model(home, visitor)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * home.size(0)
                preds = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds]
        val_accuracy = accuracy_score(val_labels, val_preds_binary)
        val_roc_auc = roc_auc_score(val_labels, val_preds)

        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f} | Train ROC-AUC: {roc_auc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")
        print("-" * 50)

# ----------------------------
# Step 7: Putting It All Together
# ----------------------------

def main():
    # ----------------------------
    # Configuration
    # ----------------------------
    output_dir = 'output'
    JSON_FILE_PATH = os.path.join(output_dir, 'player_data.json')
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 5e-4
    TEST_SIZE = 0.15  # 20% for validation
    RANDOM_STATE = 42
    MODEL_SAVE_PATH = 'dual_head_model.pth'

    # Hyperparameters for optimizer and scheduler
    WEIGHT_DECAY = 1e-2
    BETAS = (0.9, 0.95)
    MAX_GRAD_NORM = 1.0
    WARMUP_ITERS_RATIO = 0.2  # 10% of total iterations
    MIN_LR = 6e-5

    # ----------------------------
    # Check if JSON file exists
    # ----------------------------
    if not os.path.exists(JSON_FILE_PATH):
        print(f"JSON file not found at {JSON_FILE_PATH}. Please check the path.")
        return

    # ----------------------------
    # Load data
    # ----------------------------
    print("Loading JSON data...")
    data = load_json(JSON_FILE_PATH)
    print(f"Total games loaded: {len(data)}\n")

    # ----------------------------
    # Identify and Exclude First 10 Games per Team per Season
    # ----------------------------
    print("Identifying games to exclude (first 10 games per team per season)...")
    excluded_game_ids = get_excluded_game_ids(data)

    # ----------------------------
    # Preprocess data (remove excluded games and NaN)
    # ----------------------------
    print("Preprocessing data (removing excluded games and games with NaN values)...")
    processed_data = preprocess_data(data, excluded_game_ids)
    validate_labels(processed_data)
    if len(processed_data) == 0:
        print("No data available after preprocessing. Exiting.")
        return

    # ----------------------------
    # Split into training and validation sets
    # ----------------------------
    print("Splitting data into training and validation sets...")
    train_data, val_data = train_test_split(
        processed_data, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=[game['label'] for game in processed_data]
    )
    print(f"Training games: {len(train_data)} | Validation games: {len(val_data)}\n")

    # ----------------------------
    # Create Dataset and DataLoader
    # ----------------------------
    print("Creating Dataset and DataLoader...")
    train_dataset = GamesDataset(train_data, PLAYER_FEATURES)
    val_dataset = GamesDataset(val_data, PLAYER_FEATURES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------
    # Initialize model
    # ----------------------------
    num_features = len(PLAYER_FEATURES)
    model = DualHeadNetwork(num_features=num_features, player_embedding_dim=64, 
                            team_embedding_dim=128, hidden_dim=64)

    # Optionally, print model architecture
    print("Model architecture:")
    print(model)
    print("-" * 50)

    # ----------------------------
    # Train the model
    # ----------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}\n")

    train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=20, 
        lr=LEARNING_RATE, 
        device=device,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_iters_ratio=WARMUP_ITERS_RATIO,
        min_lr=MIN_LR
    )

    # ----------------------------
    # Save the model
    # ----------------------------
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
