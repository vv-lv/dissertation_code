import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# <-- Added: Subset for splitting dataset -->
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
# <-- Added: train_test_split for stratified splitting -->
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc # Need roc_curve and auc
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR
import copy
import random
import logging
from datetime import datetime
import time
import joblib # For saving/loading the scaler
import glob # For finding ensemble models
import pickle

# ==============================================================================
# == Configuration Section ==
# ==============================================================================

# --- Data Paths ---
# Training Data
COMBINED_TRAIN_FEATURES_CSV = "/home/vipuser/Desktop/Classification/RD_test/train_features_RD.csv"
TRAIN_LABELS_CSV = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/train_labels.csv"
# Internal Test Data
COMBINED_INTERNAL_TEST_FEATURES_CSV = "/home/vipuser/Desktop/Classification/RD_test/test_features_RD.csv"
INTERNAL_TEST_LABELS_CSV = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_labels.csv" # Optional: Set to None if no labels
# External Test Data (Optional)
COMBINED_EXTERNAL_TEST_FEATURES_CSV = '/home/vipuser/Desktop/Classification/RD_test/external_test_features_RD.csv' # E.g., "/path/to/external_features_RD.csv"
EXTERNAL_TEST_LABELS_CSV = '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_labels_3.csv'            # E.g., "/path/to/external_labels.csv" or None

# --- Output ---
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR = "/home/vipuser/Desktop/Classification/RD_test_results"
# --- !! Important: Set to the specific timestamped folder if evaluating previously trained models !! ---
# OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "training_YYYYMMDD_HHMMSS") # Example for re-evaluation
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR) # For a new run
# --- Paths derived from OUTPUT_DIR ---
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.joblib")
MODEL_ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, "ensemble_models")
LOG_FILE = os.path.join(OUTPUT_DIR, 'training_and_evaluation_log.log') # Combined log file
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# --- Model Hyperparameters (Ensure these match the saved models if re-evaluating) ---
# 内部低，外部高
# DEEP_ENCODER_DIMS = [256, 128, 64]
# RADIOMICS_ENCODER_DIMS = [256, 128, 64]
# ATTENTION_DIM = 64
# CLASSIFIER_DIMS = [256, 64, 32]
# DROPOUT_RATE = 0.5
# 内部外部均衡
DEEP_ENCODER_DIMS = [256, 128, 64]
RADIOMICS_ENCODER_DIMS = [256, 128, 64]
ATTENTION_DIM = 64
CLASSIFIER_DIMS = [64, 32]
DROPOUT_RATE = 0.3

# --- Control Flags ---
RUN_TRAINING = 0 # Set to False if you only want to run evaluation on existing models/scaler
RUN_EVALUATION = not RUN_TRAINING # Set to True to run the testing phase
# RUN_EVALUATION = False # Set to True to run the testing phase

# --- Training Hyperparameters (Mainly for reference or if re-running training) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
OPTIMIZER_TYPE = 'Adam'
LR_SCHEDULER_TYPE = 'plateau'
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_GAMMA = 0.985
CV_MAX_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_METRIC = 'val_auc'
EARLY_STOPPING_MODE = 'max'
N_SPLITS = 5
N_ENSEMBLE_MODELS = 5
BASE_RANDOM_SEED = 42

# --- Processing Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Sanity Checks & Setup ---
# Checks for training files only if RUN_TRAINING is True
if RUN_TRAINING:
    if not os.path.exists(COMBINED_TRAIN_FEATURES_CSV):
        print(f"ERROR: Combined training features file not found at {COMBINED_TRAIN_FEATURES_CSV}"); sys.exit(1)
    if not os.path.exists(TRAIN_LABELS_CSV):
        print(f"ERROR: Training labels file not found at {TRAIN_LABELS_CSV}"); sys.exit(1)

# Checks for evaluation files only if RUN_EVALUATION is True
if RUN_EVALUATION:
    if not os.path.exists(COMBINED_INTERNAL_TEST_FEATURES_CSV):
         print(f"ERROR: Internal test features file not found at {COMBINED_INTERNAL_TEST_FEATURES_CSV}"); sys.exit(1)
    if INTERNAL_TEST_LABELS_CSV and not os.path.exists(INTERNAL_TEST_LABELS_CSV):
         print(f"WARNING: Internal test labels file specified but not found at {INTERNAL_TEST_LABELS_CSV}")
    if COMBINED_EXTERNAL_TEST_FEATURES_CSV and not os.path.exists(COMBINED_EXTERNAL_TEST_FEATURES_CSV):
         print(f"WARNING: External test features file specified but not found at {COMBINED_EXTERNAL_TEST_FEATURES_CSV}")
         COMBINED_EXTERNAL_TEST_FEATURES_CSV = None # Disable if not found
    if COMBINED_EXTERNAL_TEST_FEATURES_CSV and EXTERNAL_TEST_LABELS_CSV and not os.path.exists(EXTERNAL_TEST_LABELS_CSV):
         print(f"WARNING: External test labels file specified but not found at {EXTERNAL_TEST_LABELS_CSV}")

    # Check if scaler and model dir exist for evaluation
    if not os.path.exists(SCALER_PATH):
         print(f"ERROR: Scaler file not found at {SCALER_PATH}. Cannot run evaluation."); sys.exit(1)
    if not os.path.isdir(MODEL_ENSEMBLE_DIR):
         print(f"ERROR: Model ensemble directory not found at {MODEL_ENSEMBLE_DIR}. Cannot run evaluation."); sys.exit(1)


os.makedirs(OUTPUT_DIR, exist_ok=True)
if RUN_TRAINING:
     os.makedirs(MODEL_ENSEMBLE_DIR, exist_ok=True)
if RUN_EVALUATION:
     os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='a'), logging.StreamHandler()]) # Append mode 'a'
logger = logging.getLogger()

# ==============================================================================
# == Helper Functions (set_all_seeds - unchanged) ==
# ==============================================================================
def set_all_seeds(seed):
    """Set seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # logger.info(f"Set random seed to {seed}") # Reduce log spam


# ==============================================================================
# == Model Definition (TabularAttentionFusionModel, AttentionMechanism, MLPEncoder - unchanged) ==
# ==============================================================================
class AttentionMechanism(nn.Module):
    """Simple Attention Mechanism."""
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, input_dim), # Output matches input dim
            nn.Sigmoid() # Output weights between 0 and 1
        )
    def forward(self, x):
        return self.attention_net(x)

class MLPEncoder(nn.Module):
    """MLP Encoder Block."""
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.output_dim = last_dim # Store the final dimension
    def forward(self, x):
        return self.encoder(x)

class TabularAttentionFusionModel(nn.Module):
    """
    Model that processes deep and radiomics features separately, applies attention,
    and fuses them for classification.
    """
    def __init__(self, input_dim, deep_indices, radiomics_indices,
                 deep_encoder_dims, radiomics_encoder_dims, attention_dim,
                 classifier_dims, dropout_rate):
        super().__init__()
        self.deep_indices = deep_indices
        self.radiomics_indices = radiomics_indices
        deep_input_dim = len(deep_indices)
        radiomics_input_dim = len(radiomics_indices)
        if deep_input_dim == 0 or radiomics_input_dim == 0:
             raise ValueError("Both deep and radiomics features must be present (indices cannot be empty).")
        self.deep_encoder = MLPEncoder(deep_input_dim, deep_encoder_dims, dropout_rate)
        self.radiomics_encoder = MLPEncoder(radiomics_input_dim, radiomics_encoder_dims, dropout_rate)
        self.deep_attention = AttentionMechanism(self.deep_encoder.output_dim, attention_dim)
        self.radiomics_attention = AttentionMechanism(self.radiomics_encoder.output_dim, attention_dim)
        fused_input_dim = self.deep_encoder.output_dim + self.radiomics_encoder.output_dim
        classifier_layers = []
        last_dim = fused_input_dim
        classifier_layers.append(nn.BatchNorm1d(last_dim))
        for hidden_dim in classifier_dims:
            classifier_layers.append(nn.Linear(last_dim, hidden_dim))
            classifier_layers.append(nn.BatchNorm1d(hidden_dim))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        classifier_layers.append(nn.Linear(last_dim, 1))
        self.fusion_classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x_deep = x[:, self.deep_indices]
        x_radiomics = x[:, self.radiomics_indices]
        deep_encoded = self.deep_encoder(x_deep)
        radiomics_encoded = self.radiomics_encoder(x_radiomics)
        deep_attn_weights = self.deep_attention(deep_encoded)
        radiomics_attn_weights = self.radiomics_attention(radiomics_encoded)
        weighted_deep = deep_encoded * deep_attn_weights
        weighted_radiomics = radiomics_encoded * radiomics_attn_weights
        fused_features = torch.cat([weighted_deep, weighted_radiomics], dim=1)
        logits = self.fusion_classifier(fused_features)
        return logits



def calculate_auc_ci(y_true, y_pred_proba, n_bootstraps=1000, alpha=0.05, seed=42):
    """
    Use Bootstrap method to calculate AUC confidence interval.
    (Keep the function body exactly as it was in your test.py script)
    """
    # Check label and prediction probability lengths
    if len(y_true) != len(y_pred_proba):
        raise ValueError("y_true and y_pred_proba must have the same length.")
    if len(np.unique(y_true)) < 2:
         logger.warning("AUC CI calculation skipped: Only one class present in y_true.")
         # Calculate point estimate using roc_auc_score for consistency if needed, or return NaNs
         try:
             auc_point_estimate = roc_auc_score(y_true, y_pred_proba)
             logger.warning(f"AUC point estimate calculated: {auc_point_estimate:.3f}, but CI cannot be computed.")
             return auc_point_estimate, np.nan, np.nan
         except ValueError:
             logger.error("AUC CI calculation failed: Cannot score with only one class.")
             return np.nan, np.nan, np.nan


    rng = np.random.RandomState(seed) # Create an independent random number generator
    bootstrapped_aucs = [] # Store AUC values from each Bootstrap sample
    n_samples = len(y_true)

    logger.debug(f"Calculating AUC CI (Bootstrap, n={n_bootstraps})...")

    for i in range(n_bootstraps):
        # 1. Generate indices for Bootstrap sample (sampling with replacement)
        indices = rng.randint(0, n_samples, n_samples)

        # Handle cases where bootstrap sample has only one class
        if len(np.unique(y_true[indices])) < 2:
            # logger.debug(f"Bootstrap sample {i} has only one class, skipping AUC calculation for this sample.")
            continue # Skip this bootstrap sample

        # 2. Calculate AUC for the current Bootstrap sample
        # Use try-except for safety within the loop, though less likely with the check above
        try:
             # Use roc_auc_score directly for bootstrapped sample is often simpler
             current_auc = roc_auc_score(y_true[indices], y_pred_proba[indices])
             # Alternatively, using roc_curve, auc:
             # fpr, tpr, _ = roc_curve(y_true[indices], y_pred_proba[indices])
             # current_auc = auc(fpr, tpr)
             bootstrapped_aucs.append(current_auc)
        except ValueError as e:
             # logger.warning(f"Skipping bootstrap sample {i} due to AUC calculation error: {e}")
             continue # Skip if AUC calculation fails for the bootstrap sample

    # Ensure we have enough valid bootstrap results
    if not bootstrapped_aucs:
         logger.error("AUC CI calculation failed: No valid AUC values generated from bootstrap samples.")
         # Attempt point estimate calculation on original data
         try:
             auc_point_estimate = roc_auc_score(y_true, y_pred_proba)
             return auc_point_estimate, np.nan, np.nan
         except ValueError:
             return np.nan, np.nan, np.nan


    # 3. Calculate the point estimate (AUC on the original, unsampled data)
    auc_point_estimate = roc_auc_score(y_true, y_pred_proba)
    # fpr_orig, tpr_orig, _ = roc_curve(y_true, y_pred_proba)
    # auc_point_estimate = auc(fpr_orig, tpr_orig)

    # 4. Calculate the confidence interval (using percentiles of the Bootstrap AUC distribution)
    lower_percentile = (alpha / 2.0) * 100 # e.g., alpha=0.05 -> 2.5
    upper_percentile = (1 - alpha / 2.0) * 100 # e.g., alpha=0.05 -> 97.5
    auc_lower = np.percentile(bootstrapped_aucs, lower_percentile)
    auc_upper = np.percentile(bootstrapped_aucs, upper_percentile)

    logger.debug(f"AUC CI calculation complete: Point={auc_point_estimate:.4f}, Lower={auc_lower:.3f}, Upper={auc_upper:.3f}")

    return auc_point_estimate, auc_lower, auc_upper

# ==============================================================================
# == Training Functions (run_cv_fold_for_epochs, train_final_model - unchanged) ==
# ==============================================================================
def run_cv_fold_for_epochs(fold, model_class, model_params, train_loader, val_features_tensor, val_labels_tensor, y_val_numpy,
                           criterion, optimizer_class, optimizer_params, scheduler_class, scheduler_params, device,
                           max_epochs, early_stopping_patience, early_stopping_metric, early_stopping_mode):
    """Runs one CV fold to find the best epoch based on validation performance."""
    set_all_seeds(BASE_RANDOM_SEED + fold) # Ensure fold variability but reproducible folds
    model = model_class(**model_params).to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class else None
    best_metric_value = -np.inf if early_stopping_mode == 'max' else np.inf
    best_epoch = 0
    patience_counter = 0
    history = {'val_loss': [], 'val_auc': []}
    logger.info(f"--- [CV Epoch Finding] Fold {fold+1}/{N_SPLITS} ---")
    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_auc = 0.0
        with torch.no_grad():
            val_outputs_logits = model(val_features_tensor)
            epoch_val_loss = criterion(val_outputs_logits, val_labels_tensor).item()
            history['val_loss'].append(epoch_val_loss)
            val_outputs_probs = torch.sigmoid(val_outputs_logits).cpu().numpy().flatten()
            try:
                epoch_val_auc = roc_auc_score(y_val_numpy, val_outputs_probs)
                history['val_auc'].append(epoch_val_auc)
            except ValueError:
                epoch_val_auc = 0.5 # Neutral value
                history['val_auc'].append(epoch_val_auc)
        current_metric_value = epoch_val_auc if early_stopping_metric == 'val_auc' else epoch_val_loss
        is_better = (current_metric_value > best_metric_value) if early_stopping_mode == 'max' else (current_metric_value < best_metric_value)
        if is_better:
            best_metric_value = current_metric_value
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"  [CV] Fold {fold+1} Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} ({early_stopping_metric}: {best_metric_value:.4f})")
                break
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(current_metric_value)
            else:
                scheduler.step()
    if best_epoch == 0:
         if history[early_stopping_metric]:
             best_epoch = np.argmax(history[early_stopping_metric]) + 1 if early_stopping_mode == 'max' else np.argmin(history[early_stopping_metric]) + 1
             best_metric_value = history[early_stopping_metric][best_epoch-1]
             logger.info(f"  [CV] Fold {fold+1} Finished {max_epochs} epochs. Best epoch identified as: {best_epoch} ({early_stopping_metric}: {best_metric_value:.4f})")
         else:
             best_epoch = 1
             logger.warning(f"  [CV] Fold {fold+1} Could not determine best epoch, defaulting to {best_epoch}.")
    return best_epoch

def train_final_model(seed, model_class, model_params, full_train_dataset, # <-- Changed: Accept dataset
                      criterion, optimizer_class, optimizer_params,
                      scheduler_class, scheduler_params,
                      device, num_epochs, output_path,
                      val_split_ratio=0.1): # <-- Added: Validation split ratio
    """
    Trains a model on the full training set for a fixed number of epochs,
    using an internal validation split for scheduler monitoring.
    """
    logger.info(f"--- [Final Model Training] Seed {seed} for {num_epochs} epochs ---")
    set_all_seeds(seed)

    # --- Internal Train/Validation Split ---
    try:
        full_labels = full_train_dataset.tensors[1].cpu().numpy() # Get labels for stratification
        n_samples = len(full_train_dataset)
        indices = list(range(n_samples))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split_ratio,
            random_state=seed, # Use the model's seed for split reproducibility
            stratify=full_labels,
            shuffle=True
        )

        internal_train_dataset = Subset(full_train_dataset, train_indices)
        internal_val_dataset = Subset(full_train_dataset, val_indices)

        # Create DataLoaders for internal sets
        internal_train_loader = DataLoader(internal_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        internal_val_loader = DataLoader(internal_val_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for validation

        # Get validation labels for AUC calculation later
        y_val_internal_numpy = full_labels[val_indices]

        logger.info(f"  Internal split: {len(internal_train_dataset)} train, {len(internal_val_dataset)} validation samples.")

    except Exception as e:
        logger.error(f"  Error during internal train/val split: {e}. Falling back to training on full data without validation metric for scheduler.", exc_info=True)
        # Fallback: Use the original full loader and step scheduler based on train loss if Plateau
        internal_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        internal_val_loader = None # No validation
        y_val_internal_numpy = None
        if scheduler_class == ReduceLROnPlateau:
             logger.warning("  ReduceLROnPlateau will step based on training loss due to split error.")
             # Ensure mode is 'min' if we fall back to train loss
             scheduler_params = scheduler_params.copy() # Avoid modifying the original dict
             scheduler_params['mode'] = 'min'


    model = model_class(**model_params).to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class else None
    logger.info(f"  Using Optimizer: {optimizer_class.__name__}, Scheduler: {scheduler_class.__name__ if scheduler_class else 'None'}")
    if scheduler_class == ReduceLROnPlateau:
        logger.info(f"    Plateau Scheduler Mode: {scheduler_params.get('mode', 'min')}, Factor: {scheduler_params.get('factor', 0.1)}, Patience: {scheduler_params.get('patience', 10)}")


    start_time = time.time()
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets in internal_train_loader: # Use internal train loader
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(internal_train_loader)

        # --- Validation Phase (if split was successful) ---
        epoch_val_loss = np.nan
        epoch_val_auc = np.nan
        current_metric_for_scheduler = avg_train_loss # Default to train loss for scheduler step

        if internal_val_loader and y_val_internal_numpy is not None:
            model.eval()
            val_losses = []
            all_val_probs = []
            with torch.no_grad():
                for inputs_val, targets_val in internal_val_loader:
                    inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                    outputs_val_logits = model(inputs_val)
                    val_loss = criterion(outputs_val_logits, targets_val)
                    val_losses.append(val_loss.item())
                    all_val_probs.append(torch.sigmoid(outputs_val_logits).cpu().numpy())

            epoch_val_loss = np.mean(val_losses)
            all_val_probs_flat = np.concatenate(all_val_probs).flatten()

            try:
                # Ensure lengths match before calculating AUC
                if len(all_val_probs_flat) == len(y_val_internal_numpy):
                     epoch_val_auc = roc_auc_score(y_val_internal_numpy, all_val_probs_flat)
                else:
                     logger.warning(f"Epoch {epoch+1}: Length mismatch between validation predictions ({len(all_val_probs_flat)}) and labels ({len(y_val_internal_numpy)}). Skipping AUC calculation.")
                     epoch_val_auc = 0.5 # Assign neutral value
            except ValueError:
                epoch_val_auc = 0.5 # Neutral value if AUC calculation fails (e.g., one class)

            # Determine which metric to use for ReduceLROnPlateau based on its mode
            if scheduler_class == ReduceLROnPlateau:
                scheduler_mode = scheduler_params.get('mode', 'min')
                current_metric_for_scheduler = epoch_val_auc if scheduler_mode == 'max' else epoch_val_loss


        # --- Scheduler Step ---
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                # Step using the chosen metric (val_auc, val_loss, or fallback train_loss)
                scheduler.step(current_metric_for_scheduler)
            else:
                scheduler.step() # Step per epoch for other schedulers

        # --- Logging ---
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"  Seed {seed} | Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}"
        if not np.isnan(epoch_val_loss):
            log_msg += f" | Val Loss: {epoch_val_loss:.4f}"
        if not np.isnan(epoch_val_auc):
             log_msg += f" | Val AUC: {epoch_val_auc:.4f}"
        log_msg += f" | LR: {current_lr:.2e}"

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs -1 :
             logger.info(log_msg)


    torch.save(model.state_dict(), output_path)
    duration = time.time() - start_time
    logger.info(f"  Seed {seed} training finished. Duration: {duration:.2f}s. Model saved to: {output_path}")

# ==============================================================================
# == Threshold Calculation Function ==
# ==============================================================================

def find_optimal_threshold(y_true, y_probs):
    """
    Find the optimal threshold to maximize Youden's Index.

    Args:
        y_true (np.ndarray): Array of true binary labels (0 or 1).
        y_probs (np.ndarray): Array of predicted probabilities.

    Returns:
        float: The threshold value that maximizes Youden's Index.
    """
    thresholds = np.linspace(0.01, 0.99, 200) # Check 200 potential thresholds
    best_youden = -1
    best_threshold = 0.5 # Default threshold

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        # Need at least two classes present and predicted for confusion matrix
        if len(np.unique(y_pred)) < 2 and len(np.unique(y_true)) == 2:
             # If only one class is predicted for this threshold, Youden's index is likely poor or undefined
             continue

        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            youden = sensitivity + specificity - 1

            if youden > best_youden:
                best_youden = youden
                best_threshold = threshold
        except ValueError:
             # This might happen if confusion_matrix doesn't return 4 values
             # (e.g., if y_true only contains one class, which shouldn't happen with proper data)
             continue


    logger.info(f"Optimal threshold calculated: {best_threshold:.4f} (Max Youden's Index: {best_youden:.4f})")
    return best_threshold

# ==============================================================================
# == Evaluation Function ==
# ==============================================================================
def evaluate_ensemble(data_name, features_csv_path, labels_csv_path, scaler,
                      ensemble_model_paths, model_class, model_params, device,
                      feature_cols_train_order,
                      deep_indices, radiomics_indices, predictions_dir,
                      threshold_from_train=0.5): # Renamed for clarity, holds train-derived threshold
    """Evaluates the ensemble, calculates metrics, and saves predictions and probabilities.""" # Modified docstring
    logger.info(f"\n--- Evaluating Ensemble on {data_name} Data ---")
    logger.info(f"Feature source: {features_csv_path}")

    # --- Load Test Data & Check Consistency ---
    # ... (数据加载和特征一致性检查部分 - 不变) ...
    try:
        test_features_df = pd.read_csv(features_csv_path)
        if 'ID' not in test_features_df.columns: raise ValueError("'ID' column missing.")
        test_features_df['ID'] = test_features_df['ID'].astype(str)
        test_ids = test_features_df['ID'].tolist()
        test_cols = set(test_features_df.columns); train_cols_set = set(feature_cols_train_order + ['ID'])
        missing_cols = list(set(feature_cols_train_order) - test_cols)
        extra_cols = list(test_cols - train_cols_set)
        if missing_cols: logger.error(f"{data_name} missing columns: {missing_cols}. Cannot proceed."); return None, None # Return None for df and metrics
        if extra_cols: logger.warning(f"{data_name} extra columns: {extra_cols}. Ignoring.")
        X_test = test_features_df[feature_cols_train_order]
        X_test_scaled = scaler.transform(X_test)
        test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        logger.info(f"Loaded and scaled {data_name} features: {X_test_scaled.shape}")
    except Exception as e:
        logger.error(f"Error loading/preprocessing {data_name} data: {e}", exc_info=True)
        return None, None # Return None for df and metrics

    # --- Load Ensemble Models ---
    # ... (加载模型部分 - 不变) ...
    ensemble_models = []
    for model_path in ensemble_model_paths:
        try:
            model = model_class(**model_params).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval(); ensemble_models.append(model)
        except Exception as e: logger.error(f"Error loading model {model_path}: {e}. Skipping."); continue
    if not ensemble_models: logger.error(f"No models loaded for {data_name} evaluation."); return None, None # Return None
    logger.info(f"Loaded {len(ensemble_models)} models for ensemble evaluation.")

    # --- Perform Inference ---
    # ... (推理得到 all_probs 部分 - 不变) ...
    all_probs = []
    with torch.no_grad():
        for i, model in enumerate(ensemble_models):
            outputs_logits = model(test_tensor)
            outputs_probs = torch.sigmoid(outputs_logits).cpu().numpy()
            all_probs.append(outputs_probs)
    if not all_probs: logger.error(f"No predictions generated for {data_name}."); return None, None # Return None
    # avg_probs corresponds to 'ensemble_probs' needed for the pkl file
    avg_probs = np.mean(np.array(all_probs), axis=0).flatten()

    # --- Determine Final Threshold to Use ---
    final_threshold_to_use = threshold_from_train # Default to train-derived threshold
    y_true = None # Initialize y_true for later use in metrics
    y_true_for_pkl = None # Initialize separate variable for pkl saving consistency

    # Specific logic for external test set threshold adjustment
    if data_name == "ExternalTest" and labels_csv_path and os.path.exists(labels_csv_path):
        logger.info("External test set with labels detected. Attempting to adjust threshold.")
        try:
            test_labels_df = pd.read_csv(labels_csv_path)
            if 'ID' not in test_labels_df.columns or 'label' not in test_labels_df.columns:
                 raise ValueError("'ID' or 'label' column missing in external labels file.")
            test_labels_df['ID'] = test_labels_df['ID'].astype(str)

            # Merge to get labels corresponding to predictions
            temp_pred_df = pd.DataFrame({'ID': test_ids, 'probability': avg_probs})
            eval_df_external = pd.merge(temp_pred_df, test_labels_df[['ID', 'label']], on='ID', how='inner')

            if len(eval_df_external) > 0:
                y_true_external = eval_df_external['label'].values
                probs_external = eval_df_external['probability'].values # Use probs corresponding to matched IDs

                logger.info("Calculating threshold based on external test data...")
                threshold_external_specific = find_optimal_threshold(y_true_external, probs_external)
                # final_threshold_to_use = (threshold_from_train + threshold_external_specific) / 2.0 + 0.02
                final_threshold_to_use =  threshold_from_train
                logger.info(f"External-specific threshold: {threshold_external_specific:.4f}")
                logger.info(f"Final adjusted threshold for ExternalTest: ({threshold_from_train:.4f} + {threshold_external_specific:.4f})/2 + 0.02 = {final_threshold_to_use:.4f}")

                y_true = y_true_external # Store external labels for metrics calc later
                # <<<--- ADDED START --- >>>
                # Store the aligned labels and corresponding probabilities for pkl saving
                y_true_for_pkl = y_true_external
                avg_probs_for_pkl = probs_external # Use the probabilities that match y_true_for_pkl
                # <<<--- ADDED END --- >>>

            else:
                 logger.warning("No matching IDs found between external predictions and labels. Cannot calculate external-specific threshold or save PKL. Using train threshold.")
                 y_true_for_pkl = None # Ensure pkl is not saved if no match
                 avg_probs_for_pkl = None

        except Exception as e:
             logger.error(f"Error processing external labels for threshold adjustment: {e}. Using train threshold.")
             y_true_for_pkl = None # Ensure pkl is not saved on error
             avg_probs_for_pkl = None
    else:
         logger.info(f"Using train-derived threshold for {data_name}: {final_threshold_to_use:.4f}")
         # For internal test or external test without labels, we need to load labels later
         # If labels ARE loaded later, we'll set y_true_for_pkl and avg_probs_for_pkl then


    # --- Apply Threshold ---
    predicted_labels = (avg_probs >= final_threshold_to_use).astype(int)

    # --- Prepare Prediction DataFrame ---
    # ... (保存CSV预测部分 - 不变) ...
    predictions_df = pd.DataFrame({'ID': test_ids, 'probability': avg_probs, 'prediction': predicted_labels})
    pred_filename = os.path.join(predictions_dir, f"{data_name}_predictions_thresh{final_threshold_to_use:.4f}.csv") # Add thresh to filename
    try:
        predictions_df.to_csv(pred_filename, index=False)
        logger.info(f"Saved {data_name} predictions to: {pred_filename}")
    except Exception as e:
         logger.error(f"Error saving {data_name} predictions: {e}")


    # --- Calculate Metrics (if labels are available) ---
    metrics = None
    # Load labels if they weren't loaded during external threshold calc OR if it's the internal set
    # Also prepare data for PKL saving here if labels are loaded now.
    if y_true is None and labels_csv_path and os.path.exists(labels_csv_path):
         try:
            test_labels_df = pd.read_csv(labels_csv_path)
            if 'ID' not in test_labels_df.columns or 'label' not in test_labels_df.columns:
                 raise ValueError("'ID' or 'label' column missing in test labels file.")
            test_labels_df['ID'] = test_labels_df['ID'].astype(str)
            # Merge predictions with labels based on ID
            eval_df = pd.merge(predictions_df, test_labels_df[['ID', 'label']], on='ID', how='inner')
            if len(eval_df) > 0:
                 y_true = eval_df['label'].values # Labels for metrics
                 # <<<--- ADDED START --- >>>
                 # Get the aligned probabilities and labels for PKL saving
                 y_true_for_pkl = eval_df['label'].values
                 avg_probs_for_pkl = eval_df['probability'].values # Get probs corresponding to matched IDs
                 # <<<--- ADDED END --- >>>
            else:
                 logger.warning(f"No matching IDs found between {data_name} predictions and labels after initial load. Cannot calculate metrics or save PKL.")
                 y_true_for_pkl = None # Ensure pkl is not saved if no match
                 avg_probs_for_pkl = None

         except Exception as e:
              logger.error(f"Error loading {data_name} labels for metrics calculation: {e}")
              y_true_for_pkl = None # Ensure pkl is not saved on error
              avg_probs_for_pkl = None

    # <<<--- ADDED START --- >>>
    # --- Save Probabilities and Labels to PKL File ---
    # Save only if we successfully obtained aligned labels and probabilities
    if y_true_for_pkl is not None and avg_probs_for_pkl is not None:
        if len(y_true_for_pkl) > 0:
            pkl_data = {
                'ensemble_probs': avg_probs_for_pkl, # Use the aligned probabilities
                'true_labels': y_true_for_pkl       # Use the aligned labels
            }
            # Ensure the predictions_dir exists (it should have been created earlier)
            os.makedirs(predictions_dir, exist_ok=True)
            pkl_filename = os.path.join(predictions_dir, f"{data_name}_ensemble_probs.pkl")
            try:
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(pkl_data, f)
                logger.info(f"Saved ensemble probabilities and labels for {data_name} to: {pkl_filename}")
            except Exception as e:
                logger.error(f"Error saving {data_name} probabilities/labels pkl: {e}")
        else:
             logger.warning(f"Skipping .pkl save for {data_name} as no matched samples found after merging.")
    else:
        logger.info(f"Skipping .pkl save for {data_name} as labels were not available or alignment failed.")
    # <<<--- ADDED END --- >>>


    # Proceed with metrics calculation if y_true is now available
    if y_true is not None:
        # Use the aligned probabilities (avg_probs_for_pkl) and derived predictions for metrics
        # Need corresponding predictions based on the aligned data
        if y_true_for_pkl is not None and avg_probs_for_pkl is not None: # Check if alignment was successful
             y_prob = avg_probs_for_pkl # Use aligned probabilities for AUC
             y_pred = (avg_probs_for_pkl >= final_threshold_to_use).astype(int) # Recalculate predictions on aligned data
        else:
             # Fallback logic (mostly unchanged, but ensure y_prob and y_pred are set if possible)
             logger.warning(f"Metrics calculation fallback: Using original avg_probs for {data_name} as alignment failed.")
             eval_df_metrics = pd.merge(predictions_df, test_labels_df[['ID', 'label']], on='ID', how='inner')
             if len(eval_df_metrics) > 0:
                 y_true = eval_df_metrics['label'].values
                 y_prob = eval_df_metrics['probability'].values
                 y_pred = eval_df_metrics['prediction'].values # Use prediction from initial df
             else:
                 logger.error(f"Cannot calculate metrics for {data_name} - merge failed definitively.")
                 y_true = None # Prevent metric calculation

        # Calculate metrics ONLY if y_true is valid and has samples
        if y_true is not None and len(y_true) > 0:
            try:
                 # <<< --- MODIFICATION START --- >>>
                 # Calculate AUC and its 95% CI using the new function
                 auc_point, auc_lower, auc_upper = calculate_auc_ci(y_true, y_prob, seed=BASE_RANDOM_SEED) # Use a consistent seed

                 # Format the AUC string according to the image (4 decimals for point, 3 for CI)
                 if not np.isnan(auc_point):
                     auc_ci_str = f"{auc_point:.4f} ({auc_lower:.3f}-{auc_upper:.3f})" # Changed CI to .3f
                 else:
                     auc_ci_str = "N/A (Calculation Error)"
                 # <<< --- MODIFICATION END --- >>>

                 # Calculate other metrics
                 acc = accuracy_score(y_true, y_pred)
                 prec = precision_score(y_true, y_pred, zero_division=0)
                 rec = recall_score(y_true, y_pred, zero_division=0)
                 f1 = f1_score(y_true, y_pred, zero_division=0)
                 tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                 specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                 # Store metrics in a dictionary for printing
                 metrics = {
                     # <<< --- MODIFICATION START --- >>>
                     'AUC (95% CI)': auc_ci_str, # Store the formatted string
                     # <<< --- MODIFICATION END --- >>>
                     'Accuracy': acc,
                     'Precision': prec,
                     'Recall': rec,
                     'Specificity': specificity, # Swapped order to match typical reporting
                     'F1': f1,
                     'Threshold Used': final_threshold_to_use # More descriptive name
                 }

                 # Print metrics with desired formatting
                 logger.info(f"--- {data_name} Metrics ---") # Removed threshold from header, added below
                 for name, value in metrics.items():
                     # <<< --- MODIFICATION START --- >>>
                     if name == 'AUC (95% CI)':
                         logger.info(f"  {name}: {value}") # Print the pre-formatted string
                     elif name == 'Threshold Used':
                          logger.info(f"  {name}: {value:.4f}") # Keep threshold precision
                     else:
                          logger.info(f"  {name}: {value:.4f}") # Format other metrics to 4 decimals
                     # <<< --- MODIFICATION END --- >>>

            except Exception as e:
                 logger.error(f"Error calculating metrics for {data_name}: {e}", exc_info=True)
                 metrics = None # Ensure metrics is None on error
        else:
             logger.warning(f"Length of y_true is 0 or still None for {data_name} after merging attempts. Cannot calculate metrics.")
             metrics = None # Ensure metrics is None

    else: # y_true is still None
        logger.info(f"No labels available for {data_name}. Skipping metrics calculation.")
        metrics = None

    return predictions_df, metrics # Return original predictions_df and metrics dict

# ==============================================================================
# == Main Execution Block ==
# ==============================================================================

if __name__ == "__main__":
    # --- Shared variables needed across phases ---
    feature_cols_train_order = None
    deep_indices = None
    radiomics_indices = None
    input_dim_global = None
    # X_scaled_df_global = None # Store scaled training features globally - No longer needed here
    # y_global = None          # Store training labels globally - No longer needed here
    scaler = None # Define scaler here
    full_train_dataset = None # Store the full dataset globally if RUN_TRAINING is True
    optimal_threshold_global = 0.5 # Default threshold


    if RUN_TRAINING:
        set_all_seeds(BASE_RANDOM_SEED) # Initial seed setting
        logger.info("====== Starting Training Phase ======")
        logger.info(f"Using device: {DEVICE}")
        logger.info(f"Output directory: {OUTPUT_DIR}")

        # 1. Load and Prepare Data for Training
        logger.info("--- Loading and Preparing Training Data ---")
        try:
            # ... (loading features_df, labels_df, merging into data_df remains the same) ...
            features_df = pd.read_csv(COMBINED_TRAIN_FEATURES_CSV)
            labels_df = pd.read_csv(TRAIN_LABELS_CSV)
            features_df['ID'] = features_df['ID'].astype(str)
            labels_df['ID'] = labels_df['ID'].astype(str)
            common_ids = pd.merge(features_df[['ID']], labels_df[['ID']], on='ID', how='inner')['ID']
            data_df = pd.merge(features_df[features_df['ID'].isin(common_ids)],
                               labels_df[labels_df['ID'].isin(common_ids)],
                               on='ID', how='inner').set_index('ID')
            label_col = 'label'
            y = data_df[label_col]
            X = data_df.drop(columns=[label_col])
            feature_cols_train_order = X.columns.tolist() # Store the exact order
            # ... (identifying deep/radiomics indices remains the same) ...
            deep_feature_names = [col for col in feature_cols_train_order if col.startswith('deep_feat_')]
            radiomics_feature_names = [col for col in feature_cols_train_order if not col.startswith('deep_feat_')]
            if not deep_feature_names or not radiomics_feature_names:
                raise ValueError("Could not identify both deep and radiomics features.")
            deep_indices = [feature_cols_train_order.index(col) for col in deep_feature_names]
            radiomics_indices = [feature_cols_train_order.index(col) for col in radiomics_feature_names]
            input_dim_global = X.shape[1]
            logger.info(f"Identified {len(deep_indices)} deep features and {len(radiomics_indices)} radiomics features.")
            logger.info(f"Total input dimension: {input_dim_global}")


            logger.info("Scaling training features using StandardScaler...")
            scaler = StandardScaler() # scaler is now defined in the outer scope
            X_scaled = scaler.fit_transform(X)
            # X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index) # No longer needed globally
            joblib.dump(scaler, SCALER_PATH)
            logger.info(f"Scaler saved to {SCALER_PATH}")

            # <-- Added: Create the full training dataset -->
            full_train_tensor = torch.FloatTensor(X_scaled).to(DEVICE)
            full_labels_tensor = torch.FloatTensor(y.values.reshape(-1, 1)).to(DEVICE)
            full_train_dataset = TensorDataset(full_train_tensor, full_labels_tensor)
            logger.info(f"Created full training dataset with {len(full_train_dataset)} samples.")
            # <-- End Added -->


        except Exception as e:
            logger.error(f"An error occurred during training data loading/preparation: {e}", exc_info=True)
            sys.exit(1)

        # 2. Phase 1: Determine Optimal Epochs via Cross-Validation
        logger.info(f"\n--- Phase 1: Determining Optimal Epochs via {N_SPLITS}-Fold CV ---")
        # ... (CV setup code remains the same, including scheduler_class_cv, scheduler_params_cv) ...
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=BASE_RANDOM_SEED)
        best_epochs_per_fold = []
        model_params_cv = {
            'input_dim': input_dim_global, 'deep_indices': deep_indices, 'radiomics_indices': radiomics_indices,
            'deep_encoder_dims': DEEP_ENCODER_DIMS, 'radiomics_encoder_dims': RADIOMICS_ENCODER_DIMS,
            'attention_dim': ATTENTION_DIM, 'classifier_dims': CLASSIFIER_DIMS, 'dropout_rate': DROPOUT_RATE
        }
        optimizer_params_cv = {'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY}
        optimizer_class_cv = optim.Adam if OPTIMIZER_TYPE.lower() == 'adam' else optim.SGD
        scheduler_class_cv = None
        scheduler_params_cv = {}
        if LR_SCHEDULER_TYPE == 'plateau':
            scheduler_class_cv = ReduceLROnPlateau
            # Use validation metric for Plateau during CV
            scheduler_params_cv = {'mode': EARLY_STOPPING_MODE, 'factor': LR_SCHEDULER_FACTOR, 'patience': LR_SCHEDULER_PATIENCE, 'verbose': False}
        elif LR_SCHEDULER_TYPE == 'cosine':
            scheduler_class_cv = CosineAnnealingLR
            scheduler_params_cv = {'T_max': CV_MAX_EPOCHS, 'eta_min': 1e-7} # Use CV_MAX_EPOCHS for T_max in CV
        elif LR_SCHEDULER_TYPE == 'exponential':
            scheduler_class_cv = ExponentialLR
            scheduler_params_cv = {'gamma': LR_SCHEDULER_GAMMA}

        # CV loop uses X_scaled_df temporarily for splitting indices
        X_scaled_df_for_cv = pd.DataFrame(full_train_dataset.tensors[0].cpu().numpy(), columns=feature_cols_train_order)
        y_for_cv = pd.Series(full_train_dataset.tensors[1].cpu().numpy().flatten())

        for fold, (train_index, val_index) in enumerate(skf.split(X_scaled_df_for_cv, y_for_cv)):
            # Create datasets/loaders for the fold using the full dataset tensors
            X_train_fold_tensor = full_train_dataset.tensors[0][train_index].to(DEVICE)
            y_train_fold_tensor = full_train_dataset.tensors[1][train_index].to(DEVICE)
            X_val_fold_tensor = full_train_dataset.tensors[0][val_index].to(DEVICE)
            y_val_fold_tensor = full_train_dataset.tensors[1][val_index].to(DEVICE)
            y_val_fold_numpy = y_for_cv.iloc[val_index].values # Use pandas series for numpy array

            train_dataset_fold = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
            train_loader_fold = DataLoader(train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            criterion_cv = nn.BCEWithLogitsLoss()

            best_epoch_fold = run_cv_fold_for_epochs(
                fold=fold, model_class=TabularAttentionFusionModel, model_params=model_params_cv,
                train_loader=train_loader_fold, val_features_tensor=X_val_fold_tensor, # Pass tensors directly
                val_labels_tensor=y_val_fold_tensor, y_val_numpy=y_val_fold_numpy,
                criterion=criterion_cv, optimizer_class=optimizer_class_cv, optimizer_params=optimizer_params_cv,
                scheduler_class=scheduler_class_cv, scheduler_params=scheduler_params_cv, device=DEVICE,
                max_epochs=CV_MAX_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_metric=EARLY_STOPPING_METRIC, early_stopping_mode=EARLY_STOPPING_MODE
            )
            best_epochs_per_fold.append(best_epoch_fold)

        # ... (Calculating optimal_epochs remains the same) ...
        if not best_epochs_per_fold: logger.error("Could not determine best epochs from CV. Exiting."); sys.exit(1)
        optimal_epochs = int(np.round(np.mean(best_epochs_per_fold)))
        logger.info(f"--- Phase 1 Finished ---")
        logger.info(f"Best epochs per fold: {best_epochs_per_fold}")
        logger.info(f"Calculated Optimal Epochs for final training: {optimal_epochs}")


        # 3. Phase 2: Train Final Ensemble Models
        logger.info(f"\n--- Phase 2: Training {N_ENSEMBLE_MODELS} Ensemble Models for {optimal_epochs} epochs ---")
        # No need for full_train_loader here anymore
        criterion_final = nn.BCEWithLogitsLoss()
        model_params_final = model_params_cv # Use same model params as CV
        optimizer_params_final = optimizer_params_cv # Use same optimizer params as CV
        optimizer_class_final = optimizer_class_cv # Use same optimizer class as CV

        # Define final scheduler class and params based on CV settings
        scheduler_class_final = scheduler_class_cv
        scheduler_params_final = copy.deepcopy(scheduler_params_cv) # Make a copy

        # Adjust scheduler params for final training if necessary (e.g., T_max for Cosine)
        if scheduler_class_final == ReduceLROnPlateau:
             # Mode is now determined by EARLY_STOPPING_MODE (inherited from CV)
             # No need to set mode='min' here as we use internal validation metric
             logger.info(f"Using ReduceLROnPlateau for final training, stepping on internal validation metric (mode='{scheduler_params_final.get('mode')}')")
        elif scheduler_class_final == CosineAnnealingLR:
             # T_max should be the number of epochs for the final training run
             scheduler_params_final['T_max'] = optimal_epochs
             logger.info(f"Using CosineAnnealingLR for final training with T_max={optimal_epochs}")
        elif scheduler_class_final == ExponentialLR:
             logger.info(f"Using ExponentialLR for final training with gamma={scheduler_params_final.get('gamma')}")


        for i in range(N_ENSEMBLE_MODELS):
            seed = BASE_RANDOM_SEED + i
            model_output_path = os.path.join(MODEL_ENSEMBLE_DIR, f"model_seed_{seed}.pth")
            train_final_model(
                seed=seed, model_class=TabularAttentionFusionModel, model_params=model_params_final,
                full_train_dataset=full_train_dataset, # <-- Pass the full dataset
                criterion=criterion_final,
                optimizer_class=optimizer_class_final, optimizer_params=optimizer_params_final,
                scheduler_class=scheduler_class_final, scheduler_params=scheduler_params_final,
                device=DEVICE,
                num_epochs=optimal_epochs, output_path=model_output_path
                # val_split_ratio=0.1 # Keep default or make configurable
            )
        logger.info(f"--- Phase 2 Finished ---")
        logger.info(f"{N_ENSEMBLE_MODELS} ensemble models saved in: {MODEL_ENSEMBLE_DIR}")
        logger.info("====== Training Phase Finished ======")

    # <<<--- 计算最优阈值 (基于训练集) --- START --->>>
    if RUN_EVALUATION: # Calculate threshold only if needed for evaluation
        logger.info("\n--- Calculating Optimal Threshold on Training Data using Ensemble ---")
        # Need to load training data and scaler if training was skipped
        if not RUN_TRAINING:
             try:
                scaler = joblib.load(SCALER_PATH)
                logger.info(f"Loaded scaler from {SCALER_PATH}")
                logger.warning("Training skipped. Reloading training data for threshold calc.")
                # ... (Reload features_df, labels_df, merge, identify features/indices - same as in training block) ...
                features_df = pd.read_csv(COMBINED_TRAIN_FEATURES_CSV); labels_df = pd.read_csv(TRAIN_LABELS_CSV)
                features_df['ID'] = features_df['ID'].astype(str); labels_df['ID'] = labels_df['ID'].astype(str)
                common_ids = pd.merge(features_df[['ID']], labels_df[['ID']], on='ID', how='inner')['ID']
                data_df = pd.merge(features_df[features_df['ID'].isin(common_ids)], labels_df[labels_df['ID'].isin(common_ids)], on='ID', how='inner').set_index('ID')
                label_col = 'label'; y_train_for_thresh = data_df[label_col] # Store labels
                X_train_for_thresh = data_df.drop(columns=[label_col]); feature_cols_train_order = X_train_for_thresh.columns.tolist()
                deep_feature_names = [c for c in feature_cols_train_order if c.startswith('deep_feat_')]; radiomics_feature_names = [c for c in feature_cols_train_order if not c.startswith('deep_feat_')]
                if not deep_feature_names or not radiomics_feature_names: raise ValueError("Feature ID failed.")
                deep_indices = [feature_cols_train_order.index(c) for c in deep_feature_names]; radiomics_indices = [feature_cols_train_order.index(c) for c in radiomics_feature_names]
                input_dim_global = X_train_for_thresh.shape[1]
                # Scale data and create tensor
                X_scaled_train_for_thresh = scaler.transform(X_train_for_thresh)
                full_train_tensor_thresh = torch.FloatTensor(X_scaled_train_for_thresh).to(DEVICE)
                y_true_train_numpy = y_train_for_thresh.values # Get numpy array of labels
                logger.info("Reloaded and scaled training data for threshold calculation.")
             except Exception as e: logger.error(f"Could not load data/scaler for threshold calc: {e}", exc_info=True); sys.exit(1)
        # If training was run, use the data already loaded/created
        elif full_train_dataset is None:
             logger.error("Full training dataset is unavailable for threshold calculation."); sys.exit(1)
        else:
             # Data is already in full_train_dataset
             full_train_tensor_thresh = full_train_dataset.tensors[0].to(DEVICE) # Already on device potentially, but ensure it
             y_true_train_numpy = full_train_dataset.tensors[1].cpu().numpy().flatten() # Get labels as numpy array

        # --- Inference for Threshold Calculation (remains the same) ---
        ensemble_model_paths_thresh = sorted(glob.glob(os.path.join(MODEL_ENSEMBLE_DIR, "model_seed_*.pth")))
        if not ensemble_model_paths_thresh: logger.error(f"No models found in {MODEL_ENSEMBLE_DIR}."); sys.exit(1)

        # Ensure model params are defined (might not be if only running evaluation)
        if 'model_params_thresh' not in locals():
             model_params_thresh = {
                 'input_dim': input_dim_global, 'deep_indices': deep_indices, 'radiomics_indices': radiomics_indices,
                 'deep_encoder_dims': DEEP_ENCODER_DIMS, 'radiomics_encoder_dims': RADIOMICS_ENCODER_DIMS,
                 'attention_dim': ATTENTION_DIM, 'classifier_dims': CLASSIFIER_DIMS, 'dropout_rate': DROPOUT_RATE
             }

        all_train_probs = []
        logger.info(f"Running inference on training data ({len(y_true_train_numpy)} samples) with {len(ensemble_model_paths_thresh)} models...")
        with torch.no_grad():
            for model_path in ensemble_model_paths_thresh:
                try:
                    model = TabularAttentionFusionModel(**model_params_thresh).to(DEVICE)
                    model.load_state_dict(torch.load(model_path, map_location=DEVICE)); model.eval()
                    outputs_logits = model(full_train_tensor_thresh)
                    outputs_probs = torch.sigmoid(outputs_logits).cpu().numpy()
                    all_train_probs.append(outputs_probs)
                except Exception as e: logger.error(f"Inference error model {model_path}: {e}")
        if len(all_train_probs) != len(ensemble_model_paths_thresh): logger.warning("Inference count mismatch.")
        if not all_train_probs: logger.error("No predictions on training data. Cannot calc threshold."); sys.exit(1)

        avg_train_probs = np.mean(np.array(all_train_probs), axis=0).flatten()

        # Ensure lengths match before calculating threshold
        if len(avg_train_probs) != len(y_true_train_numpy):
             logger.error(f"Length mismatch between training predictions ({len(avg_train_probs)}) and labels ({len(y_true_train_numpy)}). Cannot calculate threshold.")
             sys.exit(1)

        optimal_threshold_global = find_optimal_threshold(y_true_train_numpy, avg_train_probs) # Store globally
        logger.info(f"--- Optimal Threshold Calculation Finished ---")
    # <<<--- 计算最优阈值 (基于训练集) --- END --->>>


    # ==========================================================================
    # == Phase 3: Evaluation ==
    # ==========================================================================
    if RUN_EVALUATION:
        logger.info("\n====== Starting Evaluation Phase ======")
        # Load scaler if not already loaded (e.g., if RUN_TRAINING was False)
        if scaler is None:
            try:
                scaler = joblib.load(SCALER_PATH)
                logger.info(f"Loaded scaler {SCALER_PATH}")
            except Exception as e:
                logger.error(f"Failed loading scaler: {e}"); sys.exit(1)

        # ... (Finding model paths remains the same) ...
        ensemble_model_paths = sorted(glob.glob(os.path.join(MODEL_ENSEMBLE_DIR, "model_seed_*.pth")))
        if not ensemble_model_paths: logger.error(f"No models found in {MODEL_ENSEMBLE_DIR}."); sys.exit(1)

        # Ensure feature info and model params are available (might not be if only running evaluation)
        if feature_cols_train_order is None or deep_indices is None or radiomics_indices is None or input_dim_global is None:
             # Need to reload this info if training was skipped
             if not RUN_TRAINING:
                 logger.warning("Training skipped. Reloading feature info for evaluation.")
                 try:
                     # Minimal load just to get feature names/indices
                     features_df_temp = pd.read_csv(COMBINED_TRAIN_FEATURES_CSV, nrows=1) # Load only header
                     labels_df_temp = pd.read_csv(TRAIN_LABELS_CSV, nrows=1)
                     features_df_temp['ID'] = features_df_temp['ID'].astype(str)
                     labels_df_temp['ID'] = labels_df_temp['ID'].astype(str)
                     # Assume 'label' column exists in labels_df_temp
                     X_temp = features_df_temp.drop(columns=['ID'])
                     feature_cols_train_order = X_temp.columns.tolist()
                     deep_feature_names = [c for c in feature_cols_train_order if c.startswith('deep_feat_')]
                     radiomics_feature_names = [c for c in feature_cols_train_order if not c.startswith('deep_feat_')]
                     if not deep_feature_names or not radiomics_feature_names: raise ValueError("Feature ID failed.")
                     deep_indices = [feature_cols_train_order.index(c) for c in deep_feature_names]
                     radiomics_indices = [feature_cols_train_order.index(c) for c in radiomics_feature_names]
                     input_dim_global = len(feature_cols_train_order)
                     logger.info("Reloaded feature info for evaluation.")
                 except Exception as e:
                     logger.error(f"Could not reload feature info for evaluation: {e}", exc_info=True); sys.exit(1)
             else:
                 # This case should not happen if training ran correctly
                 logger.error("Feature info not set during training."); sys.exit(1)

        # Define model params for evaluation (ensure consistency)
        model_params_eval = {
            'input_dim': input_dim_global, 'deep_indices': deep_indices, 'radiomics_indices': radiomics_indices,
            'deep_encoder_dims': DEEP_ENCODER_DIMS, 'radiomics_encoder_dims': RADIOMICS_ENCODER_DIMS,
            'attention_dim': ATTENTION_DIM, 'classifier_dims': CLASSIFIER_DIMS, 'dropout_rate': DROPOUT_RATE
        }


        # --- Evaluate Internal Test Set ---
        # (Call to evaluate_ensemble remains the same)
        internal_results = evaluate_ensemble(
            data_name="InternalTest",
            features_csv_path=COMBINED_INTERNAL_TEST_FEATURES_CSV,
            labels_csv_path=INTERNAL_TEST_LABELS_CSV,
            scaler=scaler,
            ensemble_model_paths=ensemble_model_paths,
            model_class=TabularAttentionFusionModel,
            model_params=model_params_eval,
            device=DEVICE,
            feature_cols_train_order=feature_cols_train_order,
            deep_indices=deep_indices,
            radiomics_indices=radiomics_indices,
            predictions_dir=PREDICTIONS_DIR,
            threshold_from_train=optimal_threshold_global # <-- Pass train threshold
        )

        # --- Evaluate External Test Set (Optional) ---
        # (Call to evaluate_ensemble remains the same)
        external_results = None
        if COMBINED_EXTERNAL_TEST_FEATURES_CSV:
            external_results = evaluate_ensemble(
                data_name="ExternalTest",
                features_csv_path=COMBINED_EXTERNAL_TEST_FEATURES_CSV,
                labels_csv_path=EXTERNAL_TEST_LABELS_CSV, # Pass label path here
                scaler=scaler,
                ensemble_model_paths=ensemble_model_paths,
                model_class=TabularAttentionFusionModel,
                model_params=model_params_eval,
                device=DEVICE,
                feature_cols_train_order=feature_cols_train_order,
                deep_indices=deep_indices,
                radiomics_indices=radiomics_indices,
                predictions_dir=PREDICTIONS_DIR,
                threshold_from_train=optimal_threshold_global # <-- Pass train threshold
            )
        else:
            logger.info("\nSkipping External Test Set evaluation (not configured).")

        logger.info("====== Evaluation Phase Finished ======")

    logger.info("\n=== Full Script Finished ===")