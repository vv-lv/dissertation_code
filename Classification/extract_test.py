import sys
import os
import warnings # Import warnings

# 添加 Resnet.py 所在目录到系统路径
resnet_path = '/home/vipuser/Desktop/bigdata/MyProject/src'
sys.path.append(resnet_path)

# 然后就可以正常导入了
try:
    from Resnet import BinaryResNet3D_FusionModel # 替换为你要用的类/函数名
except ImportError:
    print(f"ERROR: Could not import BinaryResNet3D_FusionModel from {resnet_path}. Please ensure the path and file name are correct.")
    sys.exit(1) # Use sys.exit for cleaner termination

import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# --- MONAI Transforms ---
try:
    from monai.transforms import (
        Compose,
        SpatialPadd,
        CenterSpatialCropd,
        CropForegroundd,
        Resized
    )
except ImportError:
    print("MONAI not found. Please install it: pip install monai")
    sys.exit(1)

# --- Scikit-learn for Preprocessing ---
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from scipy.stats import spearmanr
except ImportError:
    print("Scikit-learn or SciPy not found. Please install them: pip install scikit-learn scipy")
    sys.exit(1)

# ==============================================================================
# == Configuration Section ==
# ==============================================================================

# --- Model & Feature Settings ---
MODEL_DEPTH = 18
PRETRAINED_MODEL_PATH = "/home/vipuser/Desktop/test_new_file/resnet18_20250404_192331/ensemble_models/model_seed_47/best_model.pth" # *** SET PATH TO YOUR CHECKPOINT ***

# --- Data Paths ---
RADIOMICS_DIR = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics"
# Paths to the *INITIAL* (before selection) radiomics CSV files
INITIAL_RADIOMICS_TRAIN_CSV = os.path.join(RADIOMICS_DIR, "train_features.csv") # Path to initial train features
INITIAL_RADIOMICS_TEST_CSV = os.path.join(RADIOMICS_DIR, "test_features.csv")   # Path to initial test features
# Optional: Path to an external test set CSV
EXTERNAL_RADIOMICS_TEST_CSV = os.path.join(RADIOMICS_DIR, "test_features_3.csv") # Set to path like "/path/to/your/external_test_features.csv" or leave as None

# Paths to the PREPROCESSED NIFTI files needed by the Dataset
GLOBAL_IMG_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/imagesTr'
GLOBAL_LBL_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/labelsTr'
LOCAL_IMG_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/imagesTr'
LOCAL_LBL_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTr'

GLOBAL_IMG_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/imagesTs'
GLOBAL_LBL_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/labelsTs'
LOCAL_IMG_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/imagesTs'
LOCAL_LBL_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTs'

# Optional: Paths for external test NIfTI files (if EXTERNAL_RADIOMICS_TEST_CSV is set)
EXTERNAL_GLOBAL_IMG_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/imagesTs_3' # E.g., '/path/to/external/global/images'
EXTERNAL_GLOBAL_LBL_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/labelsTs_3' # E.g., '/path/to/external/global/labels'
EXTERNAL_LOCAL_IMG_DIR  = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/imagesTs_3' # E.g., '/path/to/external/local/images'
EXTERNAL_LOCAL_LBL_DIR  = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTs_3' # E.g., '/path/to/external/local/labels'

# --- Output ---
# New directory for the final combined CSV files
OUTPUT_COMBINED_DIR = "/home/vipuser/Desktop/Classification/RD_test" # Changed output dir name
SELECTED_RADIOMICS_FEATURES_FILE = os.path.join(OUTPUT_COMBINED_DIR, "selected_radiomics_features.txt")

# --- Radiomics Preprocessing Settings ---
VARIANCE_THRESHOLD_VALUE = 0.1 # Example value, adjust as needed
CORRELATION_THRESHOLD_VALUE = 0.7 # Example value, adjust as needed

# --- Processing Settings ---
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Sanity Checks ---
if not os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"ERROR: Pretrained model checkpoint not found at {PRETRAINED_MODEL_PATH}"); sys.exit(1)
if not os.path.exists(RADIOMICS_DIR):
    print(f"ERROR: Radiomics base directory not found at {RADIOMICS_DIR}"); sys.exit(1)
if not os.path.exists(INITIAL_RADIOMICS_TRAIN_CSV):
    print(f"ERROR: Initial train radiomics CSV not found at {INITIAL_RADIOMICS_TRAIN_CSV}"); sys.exit(1)
if not os.path.exists(INITIAL_RADIOMICS_TEST_CSV):
     print(f"WARNING: Initial test radiomics CSV not found at {INITIAL_RADIOMICS_TEST_CSV}. Test set processing will be skipped.")
if EXTERNAL_RADIOMICS_TEST_CSV and not os.path.exists(EXTERNAL_RADIOMICS_TEST_CSV):
     print(f"WARNING: External test radiomics CSV specified but not found at {EXTERNAL_RADIOMICS_TEST_CSV}. External set processing will be skipped.")
     EXTERNAL_RADIOMICS_TEST_CSV = None # Disable if file not found

# Check external NIFTI paths if external CSV is valid
if EXTERNAL_RADIOMICS_TEST_CSV:
    if not all(os.path.isdir(p) for p in [EXTERNAL_GLOBAL_IMG_DIR, EXTERNAL_GLOBAL_LBL_DIR, EXTERNAL_LOCAL_IMG_DIR, EXTERNAL_LOCAL_LBL_DIR] if p):
         print(f"WARNING: External radiomics CSV is set, but one or more external NIfTI directories are missing or invalid. External set processing might fail later.")
         # Decide if you want to disable external processing here or let it fail later
         # EXTERNAL_RADIOMICS_TEST_CSV = None

os.makedirs(OUTPUT_COMBINED_DIR, exist_ok=True)

# Determine expected feature dimension
if MODEL_DEPTH in [10, 18, 34]:
    # EXPECTED_DEEP_FEAT_DIM = 512 * 2 # 1024 # Original assumption
    EXPECTED_DEEP_FEAT_DIM = 256 # Adjusted based on user script comment
elif MODEL_DEPTH in [50, 101, 152]:
    EXPECTED_DEEP_FEAT_DIM = 512 * 4 * 2 # 4096 # ResNet 50+ basic block expansion
else:
     print(f"WARNING: Unknown MODEL_DEPTH {MODEL_DEPTH}. Assuming EXPECTED_DEEP_FEAT_DIM = 256.")
     EXPECTED_DEEP_FEAT_DIM = 256

print(f"Configuration: Model Depth={MODEL_DEPTH}, Expected Deep Feature Dim={EXPECTED_DEEP_FEAT_DIM}, Device={DEVICE}")
print(f"Radiomics Preprocessing: Variance Threshold={VARIANCE_THRESHOLD_VALUE}, Correlation Threshold={CORRELATION_THRESHOLD_VALUE}")
print(f"Output directory for combined files: {OUTPUT_COMBINED_DIR}")

# ==============================================================================
# == Dataset Definition (Modified for Extraction) ==
# ==============================================================================
# (Using the MRIDataset3DFusionForExtraction class - unchanged from original)
class MRIDataset3DFusionForExtraction(Dataset):
    """
    Modified version of MRIDataset3DFusion for feature extraction.
    Returns (global_2ch, local_2ch, original_id_string).
    Applies deterministic transforms only (no random augmentations).
    """
    def __init__(
        self,
        original_ids, # Pass the original ID list directly
        global_image_paths,
        global_label_paths,
        local_image_paths,
        local_label_paths,
        is_center3=False
    ):
        super().__init__()
        # Input validation
        assert len(original_ids) == len(global_image_paths) == len(global_label_paths) == \
               len(local_image_paths) == len(local_label_paths), \
               "Input lists (IDs, paths) must have the same length."

        self.original_ids = original_ids # Store the ID list
        self.global_image_paths = global_image_paths
        self.global_label_paths = global_label_paths
        self.local_image_paths  = local_image_paths
        self.local_label_paths  = local_label_paths
        self.num_samples = len(original_ids) # Store the number of samples

        # --- Define Final Shapes ---
        self.final_shape_global = (40, 250, 250)    # center1, center2
        self.final_shape_local  = (37, 185, 250)    # center1, center2
        if is_center3:
            self.final_shape_global = (45, 300, 300)    # center3
            self.final_shape_local  = (42, 181, 252)    # center3
        # print(f"Dataset: Using global shape {self.final_shape_global}, local shape {self.final_shape_local}") # Reduce verbosity

        # --- Define Deterministic Transforms ---
        self.crop_local_transform = Compose([
            CropForegroundd(
                keys=["image_local", "label_local"], source_key="label_local",
                select_fn=lambda x: x > 0, margin=(2, 20, 20),
                allow_smaller=False, k_divisible=(1,1,1)
            ),
        ])
        self.resize_global = Compose([
            Resized(
                keys=["image_global", "label_global"], spatial_size=self.final_shape_global,
                mode=("trilinear", "nearest"), align_corners=(True, None)
            ),
        ])
        self.resize_local = Compose([
             Resized(
                 keys=["image_local", "label_local"], spatial_size=self.final_shape_local,
                 mode=("trilinear", "nearest"), align_corners=(True, None)
             ),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the original ID string for this index
        img_id_str = self.original_ids[idx]

        # ========== Load Data with Error Handling ==========
        try:
            g_path = self.global_image_paths[idx]
            # Ensure C,D,H,W order expected by MONAI/model (D,H,W -> expand C -> transpose)
            # Original code transposes H,W,D -> D,H,W then adds C=1 -> C,D,H,W
            g_data = nib.load(g_path).get_fdata(dtype=np.float32)
            g_data = np.expand_dims(np.transpose(g_data, (2,0,1)), axis=0) # D, H, W -> 1, D, H, W

            g_lbl_path = self.global_label_paths[idx]
            g_lbl_data = nib.load(g_lbl_path).get_fdata(dtype=np.float32)
            g_lbl_data = (g_lbl_data > 0).astype(np.float32)
            g_lbl_data = np.expand_dims(np.transpose(g_lbl_data, (2,0,1)), axis=0) # 1, D, H, W

            l_path = self.local_image_paths[idx]
            l_data = nib.load(l_path).get_fdata(dtype=np.float32)
            l_data = np.expand_dims(np.transpose(l_data, (2,0,1)), axis=0) # 1, D, H, W

            l_lbl_path = self.local_label_paths[idx]
            l_lbl_data = nib.load(l_lbl_path).get_fdata(dtype=np.float32)
            l_lbl_data = (l_lbl_data > 0).astype(np.float32)
            l_lbl_data = np.expand_dims(np.transpose(l_lbl_data, (2,0,1)), axis=0) # 1, D, H, W

        except Exception as e:
             print(f"ERROR loading NIfTI data for index {idx}, ID '{img_id_str}'. Error: {e}")
             # Return placeholder tensors and the ID string
             ph_g = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
             ph_g_lbl = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
             ph_l = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
             ph_l_lbl = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
             ph_g_2ch = torch.cat([ph_g, ph_g*ph_g_lbl], dim=0) # Shape (2, D, H, W)
             ph_l_2ch = torch.cat([ph_l, ph_l*ph_l_lbl], dim=0) # Shape (2, D, H, W)
             # Mark data as invalid? The extraction function handles missing features later.
             return ph_g_2ch, ph_l_2ch, img_id_str # Return ID even on error

        # ========== Construct Dict ==========
        # Ensure tensors have the correct shape (C, D, H, W) before transforms
        data_dict = {
            "image_global": torch.tensor(g_data, dtype=torch.float32), # Shape (1, D, H, W)
            "label_global": torch.tensor(g_lbl_data, dtype=torch.float32),# Shape (1, D, H, W)
            "image_local":  torch.tensor(l_data,  dtype=torch.float32), # Shape (1, D, H, W)
            "label_local":  torch.tensor(l_lbl_data, dtype=torch.float32) # Shape (1, D, H, W)
        }

        # ========== Apply Deterministic Transforms ==========
        try:
            # Apply transforms sequentially
            data_dict = self.crop_local_transform(data_dict) # Crops local image/label based on label fg
            data_dict = self.resize_global(data_dict)       # Resizes global image/label
            data_dict = self.resize_local(data_dict)        # Resizes (already cropped) local image/label
        except Exception as e:
            print(f"ERROR during MONAI transforms for index {idx}, ID '{img_id_str}'. Error: {e}")
            # Return placeholder tensors and the ID string
            ph_g = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
            ph_g_lbl = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
            ph_l = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
            ph_l_lbl = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
            ph_g_2ch = torch.cat([ph_g, ph_g*ph_g_lbl], dim=0)
            ph_l_2ch = torch.cat([ph_l, ph_l*ph_l_lbl], dim=0)
            return ph_g_2ch, ph_l_2ch, img_id_str # Return ID even on error

        # ========== Assemble 2-Channel Inputs ==========
        # Inputs should be (C=2, D, H, W)
        g_img = data_dict["image_global"] # Shape (1, D, H, W) after resize
        g_lbl = data_dict["label_global"] # Shape (1, D, H, W) after resize
        # Concatenate along the channel dimension (dim=0)
        g_2ch = torch.cat([g_img, g_img * g_lbl], dim=0) # Shape (2, D, H, W)

        l_img = data_dict["image_local"] # Shape (1, D', H', W') after crop and resize
        l_lbl = data_dict["label_local"] # Shape (1, D', H', W') after crop and resize
        l_2ch = torch.cat([l_img, l_img * l_lbl], dim=0) # Shape (2, D', H', W')

        # ========== Return Data and Original ID String ==========
        return g_2ch, l_2ch, img_id_str


# ==============================================================================
# == Hook Function and Feature Extraction Logic ==
# ==============================================================================

# --- Hook Function (Unchanged) ---
features_from_hook = []
def hook_fn(module, input_tensor, output_tensor):
    """Hook to capture the input tensor to the specified layer."""
    global features_from_hook
    tensor_to_process = None
    # Input to Linear layer is typically a tuple (tensor,)
    if isinstance(input_tensor, tuple) and len(input_tensor) > 0:
        tensor_to_process = input_tensor[0]
    elif torch.is_tensor(input_tensor):
         tensor_to_process = input_tensor # Should not happen for nn.Linear but handle just in case

    if tensor_to_process is not None:
        # Detach from graph and move to CPU *before* appending
        features = tensor_to_process.detach().cpu()
        # Append feature vectors individually to avoid large list of batches
        for i in range(features.shape[0]):
            features_from_hook.append(features[i])
    else:
        print(f"Warning: Hook received unexpected input format: {type(input_tensor)}")

# --- Feature Extraction Function (Unchanged from original) ---
def extract_deep_features_dict(model, dataloader, device, expected_feature_dim):
    """
    Extracts features using the registered hook and returns a dictionary
    mapping original ID string to the feature vector (as numpy array).
    """
    global features_from_hook
    features_from_hook = [] # Reset global list for each call

    feature_dict = {} # Store ID -> feature mapping
    processed_ids = set() # Track processed IDs to handle potential duplicates (though shuffle=False should prevent)

    model.eval() # Set model to evaluation mode
    hook_handle = None
    target_layer = None # Define target layer to attach hook

    # --- Find the layer to hook ---
    # Based on the original script, it seems to be the layer *before* the final classification
    # In BinaryResNet3D_FusionModel, this is likely the layer *before* the last nn.Linear in fusion_classifier
    # Assuming fusion_classifier is a Sequential, let's target the layer before the last one.
    # Or if the structure is [..., Pool, Flatten, Linear], target Flatten or Linear
    # The original script targeted model.fusion_classifier[2] - let's try that first.
    try:
        # Try accessing the layer directly if the structure is known
        # target_layer = model.fusion_classifier[some_index]
        # Let's assume the layer just BEFORE the final linear layer is desired.
        # If fusion_classifier is Sequential:
        if isinstance(model.fusion_classifier, nn.Sequential):
             # Find the last Linear layer index
            last_linear_idx = -1
            for i, layer in enumerate(model.fusion_classifier):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i

            if last_linear_idx > 0:
                # Hook the layer *before* the last Linear layer
                target_layer_idx = 2
                # If the layer before linear is Flatten, the *input* to Linear might be better
                # Let's stick to hooking the input of the *final* linear layer for simplicity,
                # as the user's original code targeted model.fusion_classifier[2], which might be the linear layer itself.
                target_layer = model.fusion_classifier[target_layer_idx]
                print(f"Attempting to register hook on: {type(target_layer).__name__} at index {last_linear_idx}")
                hook_handle = target_layer.register_forward_hook(hook_fn)
                # print(f"Registered forward hook on '{type(target_layer).__name__}'.") # Reduce verbosity
            elif last_linear_idx == 0: # Only one linear layer? Hook its input.
                target_layer = model.fusion_classifier[last_linear_idx]
                print(f"Attempting to register hook on the only Linear layer: {type(target_layer).__name__}")
                hook_handle = target_layer.register_forward_hook(hook_fn)
            else:
                 print("ERROR: Could not find a Linear layer in model.fusion_classifier to attach hook.")
                 return {}

        else:
             print("ERROR: model.fusion_classifier is not an nn.Sequential. Cannot automatically determine hook layer.")
             # Fallback: Try the specific index from the user's original code if needed
             # try:
             #     target_layer = model.fusion_classifier[2]
             #     print(f"Attempting to register hook on fallback index [2]: {type(target_layer).__name__}")
             #     hook_handle = target_layer.register_forward_hook(hook_fn)
             # except (IndexError, AttributeError):
             #     print("ERROR: Fallback hook on index [2] also failed.")
             #     return {}
             return {} # Fail if not Sequential

    except AttributeError:
        print("ERROR: Could not find 'fusion_classifier' attribute in the model."); return {}
    except Exception as e:
         print(f"ERROR registering hook: {e}"); return {}

    # --- Process Batches ---
    batch_start_hook_idx = 0 # Index in features_from_hook where the current batch's features start
    with torch.no_grad(): # Ensure no gradients are computed
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Deep Features")):
            try:
                # DataLoader returns (g_vol, l_vol, batch_ids_list)
                g_vol, l_vol, batch_ids_list = batch
                g_vol, l_vol = g_vol.to(device), l_vol.to(device)
            except Exception as e:
                print(f"Error unpacking/moving batch {batch_idx}: {e}. Skipping."); continue

            try:
                # Run forward pass - the hook will capture the features
                _ = model(g_vol, l_vol)
            except Exception as e:
                print(f"Error during model forward pass for batch {batch_idx}: {e}. Skipping feature assignment for this batch.")
                # Difficult to know how many features were added to hook list if forward pass fails mid-way.
                # Reset hook list pointer? Or just skip assigning for this batch. Let's skip assignment.
                # We need to know how many features *should* have been added if successful
                batch_start_hook_idx = len(features_from_hook) # Advance pointer past potentially corrupted additions
                continue

            # --- Assign features captured by the hook for THIS batch ---
            current_batch_size = len(batch_ids_list) # Number of samples in this batch
            # Expected end index in features_from_hook after this batch
            batch_end_hook_idx = batch_start_hook_idx + current_batch_size

            # Check if the hook captured the expected number of features for this batch
            if batch_end_hook_idx > len(features_from_hook):
                print(f"Warning: Hook list length mismatch after batch {batch_idx}. "
                      f"Expected end index {batch_end_hook_idx}, but hook list has {len(features_from_hook)} items. "
                      f"Assigning features for this batch may be incorrect. Skipping assignment.")
                # Advance pointer based on what's actually in the list to potentially recover for next batch
                batch_start_hook_idx = len(features_from_hook)
            else:
                # Extract the features corresponding to the current batch
                batch_features = features_from_hook[batch_start_hook_idx : batch_end_hook_idx]

                # Double-check count before assigning
                if len(batch_features) != current_batch_size:
                    print(f"Warning: Mismatch in batch {batch_idx}! Expected {current_batch_size} features, "
                          f"hook captured {len(batch_features)} features in the expected range. Skipping assignment.")
                else:
                    # Assign features to the dictionary using the ID strings
                    for i, id_str in enumerate(batch_ids_list):
                        if id_str in processed_ids:
                            print(f"Warning: Duplicate ID '{id_str}' encountered in batch {batch_idx}. Overwriting feature.")
                            # Decide whether to overwrite or skip. Overwriting might be okay if shuffle=False.

                        feature_vector = batch_features[i]
                        # Verify feature dimension before adding
                        if feature_vector.shape[0] == expected_feature_dim:
                             feature_dict[id_str] = feature_vector.numpy() # Store as numpy array
                             processed_ids.add(id_str)
                        else:
                             print(f"Warning: Incorrect feature dim ({feature_vector.shape[0]}) for ID '{id_str}'. Expected {expected_feature_dim}. Skipping this sample.")

                # Update hook list pointer for the next batch
                batch_start_hook_idx = batch_end_hook_idx


    if hook_handle:
        hook_handle.remove()
        # print("Removed forward hook.") # Reduce verbosity
    else:
        print("Warning: Hook handle was not created or already removed.")

    print(f"Deep feature extraction finished. Features extracted for {len(feature_dict)} unique IDs.")

    # Sanity check: compare extracted IDs with input dataloader IDs
    dataloader_ids = set(dataloader.dataset.original_ids)
    extracted_ids = set(feature_dict.keys())
    if dataloader_ids != extracted_ids:
        print(f"Warning: Mismatch between dataloader IDs ({len(dataloader_ids)}) and successfully extracted IDs ({len(extracted_ids)}).")
        missing_extraction = dataloader_ids - extracted_ids
        if missing_extraction:
            print(f"  - IDs in dataloader but features not extracted: {list(missing_extraction)[:5]}...") # Show a few

    return feature_dict


# ==============================================================================
# == Helper Function to Generate File Paths ==
# ==============================================================================
# (Unchanged from original)
def generate_paths_from_ids(ids, img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l):
    """Generates lists of file paths based on IDs, checking for existence."""
    g_img_paths, g_lbl_paths, l_img_paths, l_lbl_paths = [], [], [], []
    valid_ids = []
    missing_files_log = {}

    # print(f"Attempting to generate NIfTI paths for {len(ids)} IDs...") # Reduce verbosity
    for img_id in ids:
        # Try common extensions or assume .nii.gz
        found_all_for_id = False
        base_filename = str(img_id) # Ensure it's a string
        possible_filenames = [f"{base_filename}.nii.gz", f"{base_filename}.nii"] # Add more if needed

        for fname in possible_filenames:
            g_img_p = os.path.join(img_dir_g, fname)
            g_lbl_p = os.path.join(lbl_dir_g, fname)
            l_img_p = os.path.join(img_dir_l, fname)
            l_lbl_p = os.path.join(lbl_dir_l, fname)

            required_paths = [g_img_p, g_lbl_p, l_img_p, l_lbl_p]
            # Check existence for *this* filename variant
            if all(os.path.exists(p) for p in required_paths):
                g_img_paths.append(g_img_p)
                g_lbl_paths.append(g_lbl_p)
                l_img_paths.append(l_img_p)
                l_lbl_paths.append(l_lbl_p)
                valid_ids.append(img_id) # Add ID only if all files exist for this variant
                found_all_for_id = True
                break # Found files for this ID, move to next ID

        if not found_all_for_id:
            # Log missing files for the *last attempted* filename (or provide more detail)
            last_attempted_paths = [
                os.path.join(img_dir_g, possible_filenames[0]), # Log based on primary expected name
                os.path.join(lbl_dir_g, possible_filenames[0]),
                os.path.join(img_dir_l, possible_filenames[0]),
                os.path.join(lbl_dir_l, possible_filenames[0])
            ]
            missing_files_log[img_id] = [p for p in last_attempted_paths if not os.path.exists(p)]


    num_missing_ids = len(missing_files_log)
    if num_missing_ids > 0:
        print(f"WARNING: Could not find all required NIfTI files for {num_missing_ids} out of {len(ids)} IDs.")
        count = 0
        for missing_id, missing_list in missing_files_log.items():
            if count < 5: print(f"  - ID '{missing_id}': Missing files (checked variants like .nii.gz, .nii)") # More informative msg
            count += 1
        if num_missing_ids > 5: print(f"  ... and {num_missing_ids - 5} more IDs with missing files.")
        print(f"--> Proceeding with {len(valid_ids)} IDs for which all NIfTI files were found.")
    # else: # Reduce verbosity
        # print(f"Successfully found all required NIfTI files for all {len(ids)} IDs.")

    return g_img_paths, g_lbl_paths, l_img_paths, l_lbl_paths, valid_ids


# ==============================================================================
# == Radiomics Preprocessing Helper Function ==
# ==============================================================================

def preprocess_radiomics_train(df, variance_threshold, correlation_threshold):
    """
    Applies scaling, variance, and correlation filtering to training radiomics features.

    Args:
        df (pd.DataFrame): DataFrame with 'ID' column and radiomics features.
        variance_threshold (float): Threshold for VarianceThreshold selector.
        correlation_threshold (float): Threshold for Spearman correlation filtering.

    Returns:
        tuple:
            - pd.DataFrame: Filtered DataFrame with 'ID' and selected original features.
            - list: List of selected feature names.
            - StandardScaler: Fitted scaler object.
            - VarianceThreshold: Fitted variance selector object.
            - dict: Dictionary containing removed features info (optional).
    """
    print("--- Starting Radiomics Preprocessing (Train) ---")
    if 'ID' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'ID' column.")

    radiomics_cols = [col for col in df.columns if col != 'ID']
    if not radiomics_cols:
        print("WARNING: No radiomics columns found (excluding 'ID'). Skipping preprocessing.")
        return df, [], None, None, {}

    X = df[radiomics_cols]
    original_ids = df['ID']
    original_count = X.shape[1]
    print(f"Original radiomics features: {original_count}")

    # --- Handle potential non-numeric data ---
    X = X.apply(pd.to_numeric, errors='coerce')
    cols_before_drop = set(X.columns)
    X = X.dropna(axis=1, how='any') # Drop columns with any NaN after coercion
    cols_after_drop = set(X.columns)
    dropped_cols = list(cols_before_drop - cols_after_drop)
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} columns due to non-numeric values: {dropped_cols[:5]}...")
        radiomics_cols = X.columns.tolist() # Update list of columns being processed
        if not radiomics_cols:
             print("ERROR: All radiomics columns dropped due to non-numeric data. Cannot preprocess.")
             return df[['ID']], [], None, None, {'dropped_non_numeric': dropped_cols}

    # 1. Standardization (for variance and correlation calculation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=radiomics_cols, index=X.index)
    print(f"  Applied StandardScaler.")

    # 2. Variance Threshold
    var_selector = VarianceThreshold(threshold=variance_threshold)
    try:
        var_selector.fit(X_scaled_df)
        mask = var_selector.get_support()
        X_var_selected = X_scaled_df.loc[:, mask]
        selected_after_variance = X_var_selected.columns.tolist()
        removed_by_variance = X_scaled_df.columns[~mask].tolist()
        print(f"  Variance Threshold ({variance_threshold}): Kept {len(selected_after_variance)}, Removed {len(removed_by_variance)}")
        if not selected_after_variance:
            print("ERROR: Variance Threshold removed all features!")
            return df[['ID']], [], scaler, var_selector, {'removed_by_variance': removed_by_variance}
    except ValueError as e:
         print(f"ERROR during Variance Threshold fitting: {e}. Skipping variance filtering.")
         selected_after_variance = radiomics_cols # Keep all columns if fit fails
         removed_by_variance = []
         X_var_selected = X_scaled_df # Use all scaled columns for correlation

    # 3. Correlation Filtering (using Spearman on scaled data - OLD PIPELINE LOGIC)
    print(f"  Applying Correlation Filter (Spearman > {correlation_threshold}) on {len(selected_after_variance)} features using sequential removal logic...")
    # Use the scaled DataFrame filtered by variance threshold
    X_corr_input = X_scaled_df[selected_after_variance]

    # --- Start: Old Pipeline Correlation Logic ---
    selected_final = []                   # List to store features to keep
    removed_by_correlation_detail = {}    # Dictionary to store details of removed features
    remaining_features = selected_after_variance.copy() # Start with features remaining after variance filter

    # Pre-calculate correlations only if the number of features isn't excessively large
    # This can speed up if many pairwise checks are needed, but uses more memory.
    # Let's add a threshold, e.g., don't pre-calculate if > 2000 features to avoid memory issues.
    precalculated_corr = None
    if len(remaining_features) <= 2000:
         print(f"    Pre-calculating correlation matrix for {len(remaining_features)} features...")
         # Ensure calculation is done on the correct subset DataFrame
         precalculated_corr = X_corr_input.corr(method='spearman')
         print("    Correlation matrix calculation complete.")


    while len(remaining_features) > 0:
        selected_feature = remaining_features[0]  # Select the first remaining feature
        selected_final.append(selected_feature)   # Keep this feature

        highly_correlated_with_selected = [] # Features to remove in this iteration

        # Iterate through the *other* remaining features to check correlation
        for feature_to_check in remaining_features[1:]:
            correlation_value = np.nan # Default to NaN

            # Calculate or look up correlation
            try:
                if precalculated_corr is not None:
                    # Lookup from pre-calculated matrix
                    correlation_value = precalculated_corr.loc[selected_feature, feature_to_check]
                else:
                    # Calculate on the fly (using scaled data)
                    # Ensure columns exist in the scaled data frame used for input (X_corr_input)
                    if selected_feature in X_corr_input.columns and feature_to_check in X_corr_input.columns:
                        corr_result, _ = spearmanr(X_corr_input[selected_feature], X_corr_input[feature_to_check])
                        correlation_value = corr_result
                    else:
                         # This case shouldn't happen if remaining_features are columns of X_corr_input
                         print(f"Warning: Could not find {selected_feature} or {feature_to_check} in correlation input data.")

            except Exception as e:
                 print(f"    Warning: Error calculating/looking up correlation between '{selected_feature}' and '{feature_to_check}': {e}")
                 correlation_value = np.nan # Assign NaN on error

            # Check threshold (ignore NaN correlations)
            if not np.isnan(correlation_value) and abs(correlation_value) > correlation_threshold:
                highly_correlated_with_selected.append(feature_to_check)
                # Store removal reason: removed 'feature_to_check' because correlated with 'selected_feature'
                removed_by_correlation_detail[feature_to_check] = (selected_feature, correlation_value)

        # Update remaining_features: remove the current selected_feature AND those highly correlated with it
        # Create a new list for the next iteration to avoid modifying while iterating implicitly
        features_to_remove_this_round = set([selected_feature] + highly_correlated_with_selected)
        remaining_features = [f for f in remaining_features if f not in features_to_remove_this_round]

    # --- End: Old Pipeline Correlation Logic ---

    removed_by_correlation = list(removed_by_correlation_detail.keys())
    print(f"  Correlation Filter (Sequential): Kept {len(selected_final)}, Removed {len(removed_by_correlation)}")

    if not selected_final:
        print("ERROR: Correlation Filter removed all remaining features!")
        # Return ID column only from original df
        # Make sure removed_by_variance exists from the previous step
        if 'removed_by_variance' not in locals(): removed_by_variance = [] # Define if it wasn't created (e.g., variance step skipped)
        return df[['ID']], [], scaler, var_selector, {'removed_by_variance': removed_by_variance, 'removed_by_correlation': removed_by_correlation_detail}

    # 4. Prepare final DataFrame with ORIGINAL (unscaled) selected features
    final_df = df[['ID'] + selected_final]
    print(f"--- Radiomics Preprocessing (Train) Finished. Final selected features: {len(selected_final)} ---")

    results_info = {
        'dropped_non_numeric': dropped_cols,
        'removed_by_variance': removed_by_variance,
        'removed_by_correlation': removed_by_correlation_detail,
        'selected_features': selected_final
    }

    return final_df, selected_final, scaler, var_selector, results_info


def preprocess_radiomics_test(df, selected_features):
    """
    Applies feature selection based on the list derived from training data.

    Args:
        df (pd.DataFrame): Test DataFrame with 'ID' column and radiomics features.
        selected_features (list): List of feature names selected during training.

    Returns:
        pd.DataFrame: Filtered test DataFrame with 'ID' and selected features.
    """
    print("--- Applying Radiomics Preprocessing (Test) ---")
    if 'ID' not in df.columns:
        raise ValueError("Input DataFrame must contain an 'ID' column.")
    if not selected_features:
        print("WARNING: No features were selected during training. Returning only 'ID' column for test set.")
        return df[['ID']]

    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]

    if missing_features:
        print(f"  WARNING: Test set is missing {len(missing_features)} features selected during training: {missing_features[:5]}...")

    if not available_features:
        print("ERROR: Test set does not contain ANY of the features selected during training! Returning only ID column.")
        return df[['ID']]

    print(f"  Keeping {len(available_features)} features available in this test set.")
    final_df = df[['ID'] + available_features]
    print(f"--- Radiomics Preprocessing (Test) Finished ---")
    return final_df

# ==============================================================================
# == Main Execution Block ==
# ==============================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Model
    print("\n--- Loading Pretrained Model ---")
    model = BinaryResNet3D_FusionModel(model_depth=MODEL_DEPTH, freeze_branches=False) # Adapt parameters if needed
    try:
        # Robust checkpoint loading
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        # Check for common keys like 'state_dict', 'model_state_dict', or just the dict itself
        state_dict = checkpoint
        potential_keys = ['state_dict', 'model_state_dict', 'model', 'net']
        for key in potential_keys:
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break

        # Remove 'module.' prefix if saved with DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            print("  Removing 'module.' prefix from state_dict keys.")
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {PRETRAINED_MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model checkpoint file not found at {PRETRAINED_MODEL_PATH}. Exiting."); sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model weights: {e}. Exiting."); sys.exit(1)
    model.to(DEVICE)

    # --- Variables to store results from training preprocessing ---
    train_selected_radiomics_features = None
    # Scaler and VarSelector might be needed if you wanted to apply scaling to test data later,
    # but for now, we only need the selected feature names.
    # train_scaler = None
    # train_var_selector = None

    # --- Process Training Set ---
    print(f"\n=== Processing Training Set: {os.path.basename(INITIAL_RADIOMICS_TRAIN_CSV)} ===")
    train_radiomics_df_processed = None
    train_deep_feature_dict = {}

    if os.path.exists(INITIAL_RADIOMICS_TRAIN_CSV):
        try:
            # Load INITIAL training radiomics data
            initial_train_radiomics_df = pd.read_csv(INITIAL_RADIOMICS_TRAIN_CSV)
            if 'ID' not in initial_train_radiomics_df.columns:
                 print(f"ERROR: 'ID' column not found in {INITIAL_RADIOMICS_TRAIN_CSV}. Skipping training set.")
            else:
                initial_train_radiomics_df['ID'] = initial_train_radiomics_df['ID'].astype(str)
                original_train_ids = initial_train_radiomics_df['ID'].tolist()
                print(f"Loaded {len(initial_train_radiomics_df)} initial samples from {INITIAL_RADIOMICS_TRAIN_CSV}.")

                # Preprocess Radiomics (Fit and Transform)
                train_radiomics_df_processed, train_selected_radiomics_features, _, _, _ = preprocess_radiomics_train(
                    initial_train_radiomics_df.copy(), # Pass a copy
                    VARIANCE_THRESHOLD_VALUE,
                    CORRELATION_THRESHOLD_VALUE
                )

                if train_selected_radiomics_features:
                    print(f"Selected {len(train_selected_radiomics_features)} radiomics features after preprocessing.")
                    # Save the list of selected features
                    try:
                         with open(SELECTED_RADIOMICS_FEATURES_FILE, 'w') as f:
                             for feature in train_selected_radiomics_features:
                                 f.write(f"{feature}\n")
                         print(f"Saved selected feature list to: {SELECTED_RADIOMICS_FEATURES_FILE}")
                    except Exception as e:
                         print(f"Error saving selected feature list: {e}")
                else:
                    print("ERROR: No radiomics features selected during training preprocessing. Cannot proceed effectively.")
                    # Decide whether to exit or continue with empty radiomics features
                    # Let's continue for now, merge will likely fail or produce only deep features

                # Generate NIfTI paths based on ORIGINAL IDs from the initial CSV
                g_img_paths_tr, g_lbl_paths_tr, l_img_paths_tr, l_lbl_paths_tr, valid_train_ids_for_nifti = generate_paths_from_ids(
                    original_train_ids, GLOBAL_IMG_TR_DIR, GLOBAL_LBL_TR_DIR, LOCAL_IMG_TR_DIR, LOCAL_LBL_TR_DIR
                )

                if not valid_train_ids_for_nifti:
                    print("ERROR: No valid NIfTI files found for the training IDs. Cannot extract deep features.")
                    # Handle case where deep features can't be extracted
                else:
                    # Create Dataset and DataLoader for VALID IDs only for deep feature extraction
                    train_dataset = MRIDataset3DFusionForExtraction(
                        original_ids=valid_train_ids_for_nifti, # Use IDs with valid NIfTI
                        global_image_paths=g_img_paths_tr, global_label_paths=g_lbl_paths_tr,
                        local_image_paths=l_img_paths_tr, local_label_paths=l_lbl_paths_tr,
                        is_center3=False
                    )
                    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

                    # Extract deep features -> returns dict {id_str: feature_np_array}
                    train_deep_feature_dict = extract_deep_features_dict(model, train_dataloader, DEVICE, EXPECTED_DEEP_FEAT_DIM)

                    if not train_deep_feature_dict:
                        print("WARNING: Deep feature extraction for training set failed or returned empty.")
                        # Handle case where deep features are missing
                    else:
                        print(f"Successfully extracted deep features for {len(train_deep_feature_dict)} training samples.")


        except FileNotFoundError:
            print(f"ERROR: Initial train radiomics file not found: {INITIAL_RADIOMICS_TRAIN_CSV}. Skipping training set.")
        except Exception as e:
            print(f"Error processing training set: {e}. Skipping.")
            train_radiomics_df_processed = None # Ensure it's None on error

    else:
        print(f"Skipping training set processing as {INITIAL_RADIOMICS_TRAIN_CSV} not found.")


    # --- Combine and Save Training Set ---
    if train_radiomics_df_processed is not None and train_deep_feature_dict:
        print("\nCombining preprocessed radiomics and deep features for Training Set...")
        # Convert deep feature dict to DataFrame
        train_deep_df = pd.DataFrame.from_dict(train_deep_feature_dict, orient='index')
        train_deep_df.columns = [f'deep_feat_{i}' for i in range(EXPECTED_DEEP_FEAT_DIM)]
        train_deep_df.index.name = 'ID'
        train_deep_df = train_deep_df.reset_index()
        train_deep_df['ID'] = train_deep_df['ID'].astype(str) # Ensure ID type match

        # Merge PREPROCESSED radiomics with deep features
        # Use 'inner' join: only IDs present in BOTH selected radiomics AND successful deep extraction
        train_combined_df = pd.merge(train_radiomics_df_processed, train_deep_df, on='ID', how='inner')
        print(f"Combined Training DataFrame shape: {train_combined_df.shape}")
        print(f" - Kept {len(train_combined_df)} samples present in both preprocessed radiomics and successful deep feature extraction.")
        original_valid_rad_count = len(train_radiomics_df_processed)
        original_valid_deep_count = len(train_deep_df)
        print(f" - Lost {original_valid_rad_count - len(train_combined_df)} samples from preprocessed radiomics due to merge (deep feat missing).")
        print(f" - Lost {original_valid_deep_count - len(train_combined_df)} samples from deep features due to merge (radiomics missing/filtered).")


        # Save Combined Training Data
        output_train_path = os.path.join(OUTPUT_COMBINED_DIR, "train_features_RD.csv")
        try:
            train_combined_df.to_csv(output_train_path, index=False)
            print(f"Successfully saved combined training features to: {output_train_path}")
        except Exception as e:
            print(f"Error saving combined training CSV to {output_train_path}: {e}")

    elif train_radiomics_df_processed is not None and not train_deep_feature_dict:
         print("\nWARNING: Training radiomics were preprocessed, but no deep features were extracted. Saving preprocessed radiomics only.")
         output_train_path = os.path.join(OUTPUT_COMBINED_DIR, "train_features_radiomics_only.csv")
         try:
            train_radiomics_df_processed.to_csv(output_train_path, index=False)
            print(f"Successfully saved preprocessed training radiomics features (only) to: {output_train_path}")
         except Exception as e:
            print(f"Error saving preprocessed training radiomics CSV to {output_train_path}: {e}")
    else:
        print("\nSkipping saving combined training data due to errors in previous steps.")


    # --- Helper Function to Process Test Sets ---
    def process_test_set(set_name, csv_path, img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l, output_filename, IS_CENTER3):
        print(f"\n=== Processing {set_name} Set: {os.path.basename(csv_path)} ===")

        if not os.path.exists(csv_path):
            print(f"Skipping {set_name} set: CSV file not found at {csv_path}")
            return

        if not train_selected_radiomics_features:
             print(f"Skipping {set_name} set: No radiomics features were selected during training.")
             return

        # Check if NIfTI directories exist
        nifti_dirs = [img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l]
        if not all(os.path.isdir(d) for d in nifti_dirs if d): # Check if all provided paths are directories
            print(f"WARNING: Skipping {set_name} set: One or more NIfTI directories are missing or invalid: {nifti_dirs}")
            return

        test_radiomics_df_processed = None
        test_deep_feature_dict = {}

        try:
            # Load INITIAL test radiomics data
            initial_test_radiomics_df = pd.read_csv(csv_path)
            if 'ID' not in initial_test_radiomics_df.columns:
                 print(f"ERROR: 'ID' column not found in {csv_path}. Skipping {set_name} set.")
                 return
            initial_test_radiomics_df['ID'] = initial_test_radiomics_df['ID'].astype(str)
            original_test_ids = initial_test_radiomics_df['ID'].tolist()
            print(f"Loaded {len(initial_test_radiomics_df)} initial samples from {csv_path}.")

            # Apply Radiomics Preprocessing (Transform only, using features from training)
            test_radiomics_df_processed = preprocess_radiomics_test(
                initial_test_radiomics_df.copy(), # Pass a copy
                train_selected_radiomics_features # Use list from training
            )

            # Generate NIfTI paths based on ORIGINAL IDs
            g_img_paths_ts, g_lbl_paths_ts, l_img_paths_ts, l_lbl_paths_ts, valid_test_ids_for_nifti = generate_paths_from_ids(
                original_test_ids, img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l
            )

            if not valid_test_ids_for_nifti:
                print(f"WARNING: No valid NIfTI files found for the {set_name} IDs. Deep features cannot be extracted.")
                # Continue to potentially save radiomics-only data
            else:
                # Create Dataset and DataLoader for deep feature extraction
                test_dataset = MRIDataset3DFusionForExtraction(
                    original_ids=valid_test_ids_for_nifti, # Use IDs with valid NIfTI
                    global_image_paths=g_img_paths_ts, global_label_paths=g_lbl_paths_ts,
                    local_image_paths=l_img_paths_ts, local_label_paths=l_lbl_paths_ts,
                    is_center3=IS_CENTER3
                )
                test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

                # Extract deep features
                test_deep_feature_dict = extract_deep_features_dict(model, test_dataloader, DEVICE, EXPECTED_DEEP_FEAT_DIM)

                if not test_deep_feature_dict:
                    print(f"WARNING: Deep feature extraction for {set_name} set failed or returned empty.")
                else:
                    print(f"Successfully extracted deep features for {len(test_deep_feature_dict)} {set_name} samples.")

        except Exception as e:
            print(f"Error processing {set_name} set: {e}. Skipping.")
            test_radiomics_df_processed = None # Ensure it's None on error


        # --- Combine and Save Test Set ---
        if test_radiomics_df_processed is not None:
             if test_deep_feature_dict:
                 print(f"\nCombining preprocessed radiomics and deep features for {set_name} Set...")
                 test_deep_df = pd.DataFrame.from_dict(test_deep_feature_dict, orient='index')
                 test_deep_df.columns = [f'deep_feat_{i}' for i in range(EXPECTED_DEEP_FEAT_DIM)]
                 test_deep_df.index.name = 'ID'
                 test_deep_df = test_deep_df.reset_index()
                 test_deep_df['ID'] = test_deep_df['ID'].astype(str)

                 # Merge PREPROCESSED test radiomics with test deep features
                 test_combined_df = pd.merge(test_radiomics_df_processed, test_deep_df, on='ID', how='inner')
                 print(f"Combined {set_name} DataFrame shape: {test_combined_df.shape}")
                 print(f" - Kept {len(test_combined_df)} samples present in both preprocessed radiomics and successful deep feature extraction.")

                 output_test_path = os.path.join(OUTPUT_COMBINED_DIR, output_filename)
                 try:
                     test_combined_df.to_csv(output_test_path, index=False)
                     print(f"Successfully saved combined {set_name} features to: {output_test_path}")
                 except Exception as e:
                     print(f"Error saving combined {set_name} CSV to {output_test_path}: {e}")

             else:
                 # Save radiomics only if deep features failed but radiomics processed
                 print(f"\nWARNING: {set_name} radiomics were preprocessed, but no deep features were extracted. Saving preprocessed radiomics only.")
                 output_rad_only_filename = output_filename.replace(".csv", "_radiomics_only.csv")
                 output_test_path = os.path.join(OUTPUT_COMBINED_DIR, output_rad_only_filename)
                 try:
                     test_radiomics_df_processed.to_csv(output_test_path, index=False)
                     print(f"Successfully saved preprocessed {set_name} radiomics features (only) to: {output_test_path}")
                 except Exception as e:
                     print(f"Error saving preprocessed {set_name} radiomics CSV to {output_test_path}: {e}")
        else:
             print(f"\nSkipping saving combined {set_name} data due to errors in previous steps.")

    # --- Process Internal Test Set ---
    process_test_set(
        set_name="Internal Test",
        csv_path=INITIAL_RADIOMICS_TEST_CSV,
        img_dir_g=GLOBAL_IMG_TS_DIR, lbl_dir_g=GLOBAL_LBL_TS_DIR,
        img_dir_l=LOCAL_IMG_TS_DIR, lbl_dir_l=LOCAL_LBL_TS_DIR,
        output_filename="test_features_RD.csv",
        IS_CENTER3=False
    )

    # --- Process External Test Set (if configured) ---
    if EXTERNAL_RADIOMICS_TEST_CSV:
        process_test_set(
            set_name="External Test",
            csv_path=EXTERNAL_RADIOMICS_TEST_CSV,
            img_dir_g=EXTERNAL_GLOBAL_IMG_DIR, lbl_dir_g=EXTERNAL_GLOBAL_LBL_DIR,
            img_dir_l=EXTERNAL_LOCAL_IMG_DIR, lbl_dir_l=EXTERNAL_LOCAL_LBL_DIR,
            output_filename="external_test_features_RD.csv",
            IS_CENTER3=True
        )
    else:
        print("\nSkipping external test set processing as it was not configured or the CSV file was not found.")


    print("\n--- Feature extraction and combination script finished ---")