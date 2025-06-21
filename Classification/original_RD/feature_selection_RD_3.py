import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime

def set_all_seeds(seed=42):
    """Set seeds for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)

def create_visualization_dir(base_dir):
    """Create a directory for saving visualizations"""
    vis_dir = os.path.join(base_dir, f"feature_selection_vis")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def categorize_features(features):
    """
    Categorize features based on common patterns in radiomics feature names
    
    Returns:
    dict: Dictionary with categories as keys and feature counts as values
    """
    categories = {
        'Shape': 0,
        'First Order': 0,
        'GLCM': 0,
        'GLRLM': 0,
        'GLSZM': 0,
        'NGTDM': 0,
        'GLDM': 0,
        'Wavelet': 0,
        'LoG': 0,
        'Other': 0
    }
    
    pattern_mapping = {
        r'shape': 'Shape',
        r'firstorder': 'First Order',
        r'glcm': 'GLCM',
        r'glrlm': 'GLRLM',
        r'glszm': 'GLSZM',
        r'ngtdm': 'NGTDM',
        r'gldm': 'GLDM',
        r'wavelet': 'Wavelet',
        r'log': 'LoG'
    }
    
    for feature in features:
        feature_lower = feature.lower()
        categorized = False
        
        for pattern, category in pattern_mapping.items():
            if re.search(pattern, feature_lower):
                categories[category] += 1
                categorized = True
                break
                
        if not categorized:
            categories['Other'] += 1
    
    # Remove categories with zero count
    return {k: v for k, v in categories.items() if v > 0}

def plot_feature_types(selected_features, vis_dir):
    """Plot summary of feature types in the final selected features"""
    # Categorize features
    categories = categorize_features(selected_features)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(categories.keys(), categories.values(), color='skyblue')
    
    # Add count labels on top of each bar
    for i, (category, count) in enumerate(categories.items()):
        plt.text(i, count + 0.1, str(count), ha='center')
    
    plt.xlabel('Feature Category')
    plt.ylabel('Number of Features')
    plt.title('Feature Types in Final Selection')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(vis_dir, "feature_types_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved feature types summary plot")

def plot_feature_correlation_heatmap(X, selected_features, vis_dir):
    """Plot and save correlation heatmap for final selected features only"""
    # Filter dataframe to include only the selected features
    X_selected = X[selected_features]
    
    # Calculate correlation matrix
    correlation_matrix = X_selected.corr(method='spearman')
    
    plt.figure(figsize=(20, 16))
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    plt.title("Correlation Heatmap of Final Selected Features")
        
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "final_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved final correlation heatmap")

def load_selected_features(features_file):
    """
    Load previously selected features from a text file
    
    Parameters:
    -----------
    features_file : str
        Path to the file containing selected features (one feature per line)
    
    Returns:
    --------
    selected_features : list
        List of feature names
    """
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Selected features file not found: {features_file}")
    
    with open(features_file, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    
    print(f"Loaded {len(selected_features)} features from {features_file}")
    return selected_features

def apply_selected_features(original_df, selected_features):
    """Apply selected features to original data (including ID column)"""
    if 'ID' in original_df.columns:
        # Only include features that exist in the dataframe
        valid_features = [f for f in selected_features if f in original_df.columns]
        
        # If any features were missing, report it
        if len(valid_features) < len(selected_features):
            missing = set(selected_features) - set(valid_features)
            print(f"Warning: {len(missing)} features were not found in the dataset and will be ignored.")
            
        return original_df[['ID'] + valid_features]
    else:
        valid_features = [f for f in selected_features if f in original_df.columns]
        return original_df[valid_features]

def apply_features_to_dataset(data_path, selected_features, output_path):
    """
    Apply selected features to a dataset and save the result
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
    selected_features : list
        List of features to keep
    output_path : str
        Path to save the resulting dataset
    
    Returns:
    --------
    filtered_data : DataFrame
        The dataset with only the selected features
    """
    # Load dataset
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded dataset with {data.shape[0]} samples and {data.shape[1]} features")
        
        # Check if all selected features are in the dataset
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features are missing from the dataset")
            print(f"Missing features: {missing_features[:5]}..." if len(missing_features) > 5 else f"Missing features: {missing_features}")
            # Filter to only keep available features
            selected_features = [f for f in selected_features if f in data.columns]
        
        # Apply feature selection
        filtered_data = apply_selected_features(data, selected_features)
        
        # Save result
        filtered_data.to_csv(output_path, index=False)
        print(f"Filtered dataset saved to: {output_path}")
        print(f"Final dataset has {filtered_data.shape[0]} samples and {len(selected_features)} features")
        
        return filtered_data
    
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        return None

# Main program
if __name__ == "__main__":
    # Set global random seed for reproducibility
    set_all_seeds(42)
    
    # Configuration - modify these paths according to your needs
    FEATURES_FILE = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/RD/selected_features.txt"
    TEST_DATASET = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/test_features_RD_3.csv"
    OUTPUT_FILE = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics/RD/final_test_features_3.csv"
    
    print("=== Feature Application Mode ===")
    print(f"Loading pre-selected features from: {FEATURES_FILE}")
    
    # Load pre-selected features
    selected_features = load_selected_features(FEATURES_FILE)
    
    # Apply to test dataset
    print(f"\nApplying features to test dataset: {TEST_DATASET}")
    apply_features_to_dataset(TEST_DATASET, selected_features, OUTPUT_FILE)
    
    print("\nAll processing completed!")