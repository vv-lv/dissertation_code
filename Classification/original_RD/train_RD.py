import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Use ParameterGrid for cleaner param combination generation
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
import torch
import pickle
import logging
from datetime import datetime

# Assuming models.py contains create_model, LogisticRegressionModel, SVMModel, ANNModel
# Make sure models.py is in the same directory or Python path
try:
    from models import create_model, LogisticRegressionModel, SVMModel, ANNModel
except ImportError:
    print("Error: Could not import from models.py. Make sure it's in the correct path.")
    exit()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train')

def load_data(features_path, labels_path):
    """
    Load and prepare data.

    Args:
        features_path (str): Path to the features CSV file.
        labels_path (str): Path to the labels CSV file.

    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the label Series.
    """
    logger.info(f"Loading data - Features: {features_path}, Labels: {labels_path}")
    try:
        features_df = pd.read_csv(features_path)
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}")
        raise

    # Merge features and labels based on 'ID'
    # Ensure 'ID' column exists in both files
    if 'ID' not in features_df.columns or 'ID' not in labels_df.columns:
        logger.error("Missing 'ID' column in feature or label file for merging.")
        raise ValueError("Missing 'ID' column for merging.")

    df = pd.merge(features_df, labels_df, on='ID')

    # Identify potential non-feature columns (adjust if needed)
    non_feature_cols = ['ID', 'label']
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    X = df[feature_cols]
    y = df['label']

    logger.info(f"Data loaded successfully - Samples: {X.shape[0]}, Features: {X.shape[1]}")
    logger.info(f"Label distribution:\n{y.value_counts()}")

    # Check for missing values
    if X.isnull().values.any():
        logger.warning("Missing values detected in features! Consider imputation.")
        # Example: Impute with mean (or use more sophisticated methods)
        # X = X.fillna(X.mean())
    if y.isnull().values.any():
        logger.warning("Missing values detected in labels! Dropping rows with missing labels.")
        original_count = len(df)
        df = df.dropna(subset=['label'])
        X = df[feature_cols]
        y = df['label']
        logger.info(f"Dropped {original_count - len(df)} rows due to missing labels.")


    return X, y


def preprocess_data(X_train, X_val=None):
    """
    Preprocess data using StandardScaler.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        X_val (pd.DataFrame or np.ndarray, optional): Validation features. Defaults to None.

    Returns:
        tuple: Contains scaled training data, scaled validation data (if provided),
               and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler

    return X_train_scaled, scaler


def calculate_youden_index(fpr, tpr, thresholds):
    """
    Calculate the best threshold based on Youden's J statistic.

    Args:
        fpr (np.ndarray): False positive rates.
        tpr (np.ndarray): True positive rates.
        thresholds (np.ndarray): Corresponding thresholds.

    Returns:
        tuple: (best_threshold, best_youden_index)
    """
    youden_index = tpr - fpr
    # Handle cases where multiple thresholds give the same max Youden index; pick the first one
    max_youden_idx = np.argmax(youden_index)
    best_threshold = thresholds[max_youden_idx]
    best_youden = youden_index[max_youden_idx]

    # Handle potential edge case where threshold might be infinite
    if np.isinf(best_threshold):
         # Find the closest non-infinite threshold if the optimal one is inf
         non_inf_indices = np.where(~np.isinf(thresholds))[0]
         if len(non_inf_indices) > 0:
             closest_idx_to_max = non_inf_indices[np.argmin(np.abs(non_inf_indices - max_youden_idx))]
             best_threshold = thresholds[closest_idx_to_max]
             logger.warning(f"Infinite optimal threshold found. Using closest finite threshold: {best_threshold:.4f}")
         else:
              best_threshold = 0.5 # Fallback if all thresholds are somehow infinite
              logger.error("All thresholds are infinite. Falling back to 0.5 threshold.")


    return best_threshold, best_youden


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate various classification metrics based on predicted probabilities and a threshold.

    Args:
        y_true (np.ndarray or pd.Series): True binary labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        threshold (float, optional): Classification threshold. Defaults to 0.5.

    Returns:
        dict: A dictionary containing Accuracy, Sensitivity, Specificity, PPV, NPV, and Threshold.
    """
    # Ensure y_pred_proba is 1D
    if y_pred_proba.ndim > 1:
        logger.warning(f"y_pred_proba has shape {y_pred_proba.shape}, expected 1D. Taking the second column.")
        if y_pred_proba.shape[1] >= 2:
            y_pred_proba = y_pred_proba[:, 1]
        else: # Handle case of single column output
             y_pred_proba = y_pred_proba.flatten()


    y_pred = (y_pred_proba >= threshold).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # Handle cases where only one class is present in y_true or y_pred
        logger.warning(f"Could not unpack confusion matrix. Class distribution might be skewed or predictions constant. y_true unique: {np.unique(y_true)}, y_pred unique: {np.unique(y_pred)}")
        # Provide default/zero values for metrics in this case
        return {
            'Accuracy': accuracy_score(y_true, y_pred), # Accuracy might still be meaningful
            'Sensitivity': 0, 'Specificity': 0, 'PPV': 0, 'NPV': 0,
            'Threshold': threshold
        }


    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0          # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # Negative Predictive Value

    metrics = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Threshold': threshold
    }

    return metrics


def cross_validate_model(model_type, X, y, param_grid, cv=5, results_dir=None):
    """
    Perform stratified cross-validation to evaluate model parameters,
    train the final model on all data using best parameters, and save results.
    Uses ParameterGrid for parameter combinations.
    Handles ANN internal early stopping using validation folds.

    Args:
        model_type (str): Type of model ('LR', 'SVM', 'ANN').
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        param_grid (dict): Dictionary defining the parameter grid for the model.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.
        results_dir (str, optional): Directory to save results (model, scaler, plots). Defaults to None.

    Returns:
        tuple: (best_model, best_params, cv_results_df)
               - best_model: The final model trained on all data with best parameters.
               - best_params: The dictionary of best parameters found.
               - cv_results_df: DataFrame containing detailed results for each fold and parameter combination.
    """
    logger.info(f"Starting {cv}-fold cross-validation for {model_type}...")

    # Generate all parameter combinations
    parameter_iterator = ParameterGrid(param_grid)
    all_param_combinations = list(parameter_iterator)
    if not all_param_combinations:
        logger.error("Parameter grid is empty!")
        return None, None, None
    logger.info(f"Total parameter combinations to evaluate: {len(all_param_combinations)}")

    # --- Initialize results structure dynamically ---
    cv_results = {
        'fold_idx': [], 'auc': [], 'best_threshold': [],
        'accuracy': [], 'sensitivity': [], 'specificity': [],
        'ppv': [], 'npv':[] # Added PPV/NPV
    }
    # Find all unique parameter keys across all combinations
    all_possible_param_keys = set()
    for params in all_param_combinations:
        all_possible_param_keys.update(params.keys())
    # Add keys to cv_results
    for param_name in sorted(list(all_possible_param_keys)):
        cv_results[param_name] = []
    # ---------------------------------------------

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    param_performance = [] # Stores {'params': ..., 'mean_auc': ...}

    # --- Loop through parameter combinations ---
    for params in all_param_combinations:
        logger.info(f"Evaluating parameters: {params}")
        fold_aucs = []
        fold_thresholds = [] # Store thresholds from each fold for averaging later if needed

        # --- Loop through CV folds ---
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Preprocess data (scaler fitted only on train fold)
            X_train_scaled, X_val_scaled, fold_scaler = preprocess_data(X_train_fold, X_val_fold)

            # Prepare parameters for model creation
            current_params = params.copy()
            if model_type.upper() == 'ANN':
                current_params['input_dim'] = X_train_scaled.shape[1] # Add input_dim for ANN

            # Create model instance
            # Wrap in try-except for robustness against bad parameter combinations
            try:
                model = create_model(model_type, **current_params)

                # --- Train model ---
                if model_type.upper() == 'ANN':
                    # ANNModel uses the validation fold for internal early stopping
                    logger.debug(f"  Fold {fold_idx+1}: Training ANN with internal validation...")
                    # Ensure labels are numpy arrays for ANN fit method if it expects them
                    model.fit(X_train_scaled, y_train_fold.to_numpy(),
                              X_val=X_val_scaled, y_val=y_val_fold.to_numpy())
                else:
                    # Standard models train on the training fold only
                    model.fit(X_train_scaled, y_train_fold)
                # -----------------

                # --- Evaluate on validation fold ---
                y_val_proba = model.predict_proba(X_val_scaled)
                # Ensure y_val_proba is 1D array of positive class probabilities
                if y_val_proba.ndim == 2:
                     if y_val_proba.shape[1] >= 2:
                         y_val_proba = y_val_proba[:, 1]
                     else: # Handle single column output case
                          y_val_proba = y_val_proba.flatten()


                # Calculate ROC curve and AUC
                fpr, tpr, thresholds_roc = roc_curve(y_val_fold, y_val_proba)
                fold_auc = auc(fpr, tpr)

                # Calculate best threshold (Youden index) for this fold
                best_threshold_fold, _ = calculate_youden_index(fpr, tpr, thresholds_roc)
                fold_thresholds.append(best_threshold_fold) # Store fold threshold

                # Calculate metrics using the fold-specific best threshold
                metrics = calculate_metrics(y_val_fold, y_val_proba, threshold=best_threshold_fold)

                # --- Store results for this fold ---
                fold_aucs.append(fold_auc)
                cv_results['fold_idx'].append(fold_idx + 1) # Use 1-based index for reporting
                cv_results['auc'].append(fold_auc)
                cv_results['best_threshold'].append(best_threshold_fold)
                cv_results['accuracy'].append(metrics['Accuracy'])
                cv_results['sensitivity'].append(metrics['Sensitivity'])
                cv_results['specificity'].append(metrics['Specificity'])
                cv_results['ppv'].append(metrics['PPV'])
                cv_results['npv'].append(metrics['NPV'])

                # Record the parameters used for this fold
                for param_name in all_possible_param_keys:
                    cv_results[param_name].append(params.get(param_name)) # Use get for safety
                # -----------------------------------

                logger.info(f"  Fold {fold_idx+1}/{cv} - Val AUC: {fold_auc:.4f}, Best Threshold: {best_threshold_fold:.4f}, Acc: {metrics['Accuracy']:.4f}")

            except Exception as e:
                 logger.error(f"  Fold {fold_idx+1}/{cv} - Error during training/evaluation with params {params}: {e}", exc_info=False) # Set exc_info=True for full traceback
                 # Append NaN or skip appending for this fold if an error occurs? Appending NaN helps keep structure.
                 fold_aucs.append(np.nan)
                 cv_results['fold_idx'].append(fold_idx + 1)
                 cv_results['auc'].append(np.nan)
                 cv_results['best_threshold'].append(np.nan)
                 cv_results['accuracy'].append(np.nan)
                 cv_results['sensitivity'].append(np.nan)
                 cv_results['specificity'].append(np.nan)
                 cv_results['ppv'].append(np.nan)
                 cv_results['npv'].append(np.nan)
                 for param_name in all_possible_param_keys:
                     cv_results[param_name].append(params.get(param_name)) # Still record params
                 # Optionally break inner loop or continue to next fold
                 continue # Continue to next fold

        # --- Post-fold processing for current parameters ---
        # Calculate average performance, ignoring NaNs from errors
        mean_auc = np.nanmean(fold_aucs) if fold_aucs else 0
        mean_threshold = np.nanmean(fold_thresholds) if fold_thresholds else 0.5 # Avg threshold across folds
        param_performance.append({'params': params, 'mean_auc': mean_auc, 'mean_threshold': mean_threshold})
        logger.info(f"Parameters {params} - Average Val AUC across {cv} folds: {mean_auc:.4f}")
        # --------------------------------------------------

    # --- Find best parameters based on mean validation AUC ---
    if not param_performance:
         logger.error("No parameter combinations were successfully evaluated.")
         return None, None, None

    # Sort by mean_auc descending, handling potential NaN mean_auc values (treat NaN as worst)
    best_performance = max(param_performance, key=lambda x: x['mean_auc'] if not np.isnan(x['mean_auc']) else -np.inf)

    best_params = best_performance['params']
    best_mean_auc = best_performance['mean_auc']
    # Consider using the mean threshold from the best param set's CV folds?
    # best_threshold_from_cv = best_performance['mean_threshold']
    logger.info(f"Cross-validation finished. Best Mean Val AUC: {best_mean_auc:.4f} with params: {best_params}")
    # logger.info(f"Average threshold from CV for best params: {best_threshold_from_cv:.4f}")

    # Convert results to DataFrame
    cv_results_df = pd.DataFrame(cv_results)

    # --- Train final model using best parameters on ALL data ---
    logger.info(f"Training final {model_type} model using best parameters on the entire dataset...")

    # Preprocess the entire dataset using a new scaler fitted on all data
    # Note: This is standard practice, but assumes the test set will be scaled similarly.
    # Alternatively, one could argue to keep the scaler from one of the CV folds, but fitting on all data is common.
    X_scaled, final_scaler = preprocess_data(X)
    y_np = y.to_numpy() # Ensure numpy array for consistency

    # Prepare final parameters
    final_best_params = best_params.copy()
    if model_type.upper() == 'ANN':
        final_best_params['input_dim'] = X_scaled.shape[1]

    # Create the final model
    best_model = create_model(model_type, **final_best_params)

    # Train the final model
    if model_type.upper() == 'ANN':
        # For the final ANN, use a small internal validation split for its early stopping mechanism
        logger.info("Creating internal validation split (10%) for final ANN model training (early stopping)...")
        X_final_train, X_final_val, y_final_train, y_final_val = train_test_split(
            X_scaled, y_np, test_size=0.1, random_state=42, stratify=y_np
        )
        best_model.fit(X_final_train, y_final_train, X_val=X_final_val, y_val=y_final_val)
    else:
        # Train standard models on the full dataset
        best_model.fit(X_scaled, y_np)

    logger.info("Final model training complete.")

    # --- Evaluate final model on the *entire training set* (for reference) ---
    train_proba = best_model.predict_proba(X_scaled)
    if train_proba.ndim == 2: # Ensure 1D
         if train_proba.shape[1] >= 2:
             train_proba = train_proba[:, 1]
         else:
              train_proba = train_proba.flatten()


    fpr_train, tpr_train, thresholds_train = roc_curve(y_np, train_proba)
    train_auc = auc(fpr_train, tpr_train)

    # Calculate the optimal threshold on the *entire training set* predictions
    # This is the threshold to be saved and potentially used later on test data
    final_best_threshold, _ = calculate_youden_index(fpr_train, tpr_train, thresholds_train)
    logger.info(f"Final threshold calculated on *entire* training set predictions: {final_best_threshold:.4f}")

    # Calculate metrics on the training set using this final threshold
    train_metrics = calculate_metrics(y_np, train_proba, threshold=final_best_threshold)

    logger.info(f"Final model performance on *entire* training set:")
    logger.info(f"  AUC: {train_auc:.4f}")
    logger.info(f"  Metrics at threshold {final_best_threshold:.4f}:")
    for metric, value in train_metrics.items():
        if metric != 'Threshold':
            logger.info(f"    {metric}: {value:.4f}")

    # --- Save results if directory provided ---
    if results_dir:
        try:
            os.makedirs(results_dir, exist_ok=True)

            # --- 修改开始 ---
            # 移除时间戳生成
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # <-- 删除或注释掉这行

            # 使用固定的基础文件名 (仅基于模型类型)
            # base_filename = f"{model_type.lower()}_{timestamp}" # <-- 删除或注释掉这行
            base_filename = f"{model_type.lower()}"               # <-- 修改为此行
            # --- 修改结束 ---


            # 保存最终模型 (现在使用固定名称)
            model_path = os.path.join(results_dir, f"{base_filename}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)

            # 保存最终的 scaler (现在使用固定名称)
            scaler_path = os.path.join(results_dir, f"{base_filename}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(final_scaler, f)

            # 保存最终的阈值 (现在使用固定名称)
            threshold_path = os.path.join(results_dir, f"{base_filename}_threshold.pkl")
            with open(threshold_path, 'wb') as f:
                pickle.dump(final_best_threshold, f)

            # 保存详细的交叉验证结果 (现在使用固定名称)
            cv_results_path = os.path.join(results_dir, f"{base_filename}_cv_results.csv")
            cv_results_df.to_csv(cv_results_path, index=False)

            # 保存 ROC 曲线图 (现在使用固定名称)
            plt.figure(figsize=(8, 6))
            # ... (绘图代码不变) ...
            roc_save_path = os.path.join(results_dir, f"{base_filename}_train_roc_curve.png") # <-- 文件名也已更新
            plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
            plt.close() # Close the plot

            # 日志信息现在可以更简洁
            logger.info(f"Final model, scaler, threshold, CV results, and ROC plot saved to {results_dir} (using fixed names like {base_filename}_*)")

        except Exception as e:
            logger.error(f"Error saving results to {results_dir}: {e}", exc_info=True)

    # Return the trained model, best parameters, and CV results
    return best_model, best_params, cv_results_df


def main():
    # --- Configuration ---
    # IMPORTANT: Update these paths to your actual file locations
    config = {
        'features_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/RD/final_train_features.csv', # <--- CHANGE THIS
        'labels_path': '/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/train_labels.csv',       # <--- CHANGE THIS
        'results_dir': '/home/vipuser/Desktop/Classification/train_results_RD',                # Directory to save results
        'models_to_train': ['LR', 'SVM', 'ANN'],             # Models to include
        'random_state': 42                                   # Global random seed
    }

    # Create base results directory
    base_results_dir = config['results_dir']
    os.makedirs(base_results_dir, exist_ok=True)

    # --- Parameter Grids ---
    param_grids = {
        'LR': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear'], # Good solver for smaller datasets, supports L1/L2
            'max_iter': [1000],      # Increased max_iter
            'random_state': [config['random_state']]
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0, 50.0], # Expanded C range
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'], # Relevant for 'rbf' kernel
            'probability': [True],      # Needed for predict_proba
            'random_state': [config['random_state']]
        },
        'ANN': {
            # Structure
            'hidden_dim1': [32, 64],
            'hidden_dim2': [16, 32],
            'dropout_rate': [0.3, 0.5],
            # Optimizer & Regularization
            'learning_rate': [0.001, 0.005],
            'weight_decay': [1e-5, 1e-4, 1e-3], # L2 regularization
            # Training Control
            'batch_size': [32, 64],           # Test different batch sizes
            'num_epochs': [150],             # Max epochs (early stopping likely triggers earlier)
            'early_stopping_patience': [20], # Increased patience
            # LR Scheduler (using defaults/fixed values for patience/T_max/gamma inside ANNModel)
            'lr_scheduler_type': ['plateau', 'cosine'],
            # Other
            'random_state': [config['random_state']]
            # Note: 'input_dim' is added dynamically in cross_validate_model
        }
    }

    # --- Load Data ---
    try:
        X, y = load_data(config['features_path'], config['labels_path'])
    except Exception as e:
        logger.error(f"Failed to load data. Exiting. Error: {e}")
        return

    # --- Train and Evaluate Models ---
    all_results = {}
    for model_type in config['models_to_train']:
        if model_type not in param_grids:
            logger.warning(f"No parameter grid defined for model type '{model_type}'. Skipping.")
            continue

        logger.info(f"===== Starting Training for {model_type} =====")

        # Create a dedicated subdirectory for this model's results
        model_results_dir = os.path.join(base_results_dir, model_type)
        os.makedirs(model_results_dir, exist_ok=True)

        # Run cross-validation and final training
        best_model, best_params, cv_results_df = cross_validate_model(
            model_type=model_type,
            X=X,
            y=y,
            param_grid=param_grids[model_type],
            cv=5, # Number of CV folds
            results_dir=model_results_dir # Pass the specific dir
        )

        if best_model is not None:
            all_results[model_type] = {
                'best_model': best_model, # Keep reference if needed later
                'best_params': best_params,
                'cv_results_df': cv_results_df
            }
            # Calculate avg CV AUC from the results dataframe
            # Handle potential NaNs if some folds failed
            avg_cv_auc = cv_results_df['auc'].mean() if not cv_results_df['auc'].isnull().all() else np.nan

            logger.info(f"===== {model_type} Training Summary =====")
            logger.info(f"  Best Parameters: {best_params}")
            logger.info(f"  Average Cross-Validation AUC: {avg_cv_auc:.4f}")
            logger.info(f"  Results saved in: {model_results_dir}")
        else:
            logger.error(f"Failed to train model {model_type}.")

        logger.info(f"===== Finished Training for {model_type} =====")


    # --- Generate Final Summary ---
    summary_data = []
    for model_type, results in all_results.items():
        # Recalculate avg_cv_auc safely from the dataframe
        cv_df = results['cv_results_df']
        avg_cv_auc = cv_df['auc'].mean() if not cv_df['auc'].isnull().all() else np.nan

        summary_data.append({
            'Model Type': model_type,
            'Average CV AUC': f"{avg_cv_auc:.4f}" if not np.isnan(avg_cv_auc) else "N/A",
            'Best Parameters': str(results['best_params']) # Convert dict to string for CSV
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(base_results_dir, "overall_training_summary.csv")
        try:
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Overall training summary saved to: {summary_path}")
            print("\n--- Overall Training Summary ---")
            print(summary_df.to_string(index=False))
            print("------------------------------")
        except Exception as e:
            logger.error(f"Failed to save overall summary: {e}")
    else:
        logger.warning("No models were successfully trained. No summary generated.")

    logger.info("Script finished.")


if __name__ == "__main__":
    main()