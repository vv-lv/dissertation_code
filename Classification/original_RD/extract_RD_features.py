import sys
import os

# 添加 Resnet.py 所在目录到系统路径
resnet_path = '/home/vipuser/Desktop/bigdata/MyProject/src'
sys.path.append(resnet_path)

# 然后就可以正常导入了
from Resnet import BinaryResNet3D_FusionModel  # 替换为你要用的类/函数名

# -*- coding: utf-8 -*-
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd # Import pandas
import os
import glob
from tqdm import tqdm

# --- MONAI Transforms ---
try:
    from monai.transforms import (
        Compose,
        # Augmentations removed as they are not used for extraction
        SpatialPadd,
        CenterSpatialCropd,
        CropForegroundd,
        Resized
    )
except ImportError:
    print("MONAI not found. Please install it: pip install monai")
    exit()

# ==============================================================================
# == Configuration Section ==
# ==============================================================================

# --- Model & Feature Settings ---
MODEL_DEPTH = 18
PRETRAINED_MODEL_PATH = "/home/vipuser/Desktop/test_new_file/resnet18_20250404_192331/ensemble_models/model_seed_47/best_model.pth" # *** SET PATH TO YOUR CHECKPOINT ***

# --- Data Paths ---
RADIOMICS_DIR = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics"
# Paths to the *EXISTING SELECTED* radiomics CSV files
EXISTING_RADIOMICS_TRAIN_CSV = os.path.join(RADIOMICS_DIR, "radiomics", "final_train_features.csv")
EXISTING_RADIOMICS_TEST_CSV = os.path.join(RADIOMICS_DIR, "radiomics", "final_test_features.csv")

# Paths to the PREPROCESSED NIFTI files needed by the Dataset
GLOBAL_IMG_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/imagesTr'
GLOBAL_LBL_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/labelsTr'
LOCAL_IMG_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/imagesTr'
LOCAL_LBL_TR_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTr'

GLOBAL_IMG_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/imagesTs'
GLOBAL_LBL_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62/labelsTs'
LOCAL_IMG_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/imagesTs'
LOCAL_LBL_TS_DIR = '/home/vipuser/Desktop/Data/Task02_PASp62_edge/labelsTs'

# --- Output ---
# New directory for the final combined CSV files
OUTPUT_COMBINED_DIR = "/home/vipuser/Desktop/Data/Task02_PASp62_radiomics"

# --- Processing Settings ---
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CENTER3 = False

# --- Sanity Checks ---
if not os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"ERROR: Pretrained model checkpoint not found at {PRETRAINED_MODEL_PATH}"); exit()
if not os.path.exists(RADIOMICS_DIR):
    print(f"ERROR: Radiomics base directory not found at {RADIOMICS_DIR}"); exit()
if not os.path.exists(EXISTING_RADIOMICS_TRAIN_CSV):
    print(f"ERROR: Existing selected train radiomics CSV not found at {EXISTING_RADIOMICS_TRAIN_CSV}"); exit()
# Optional check for test file if needed later
# if not os.path.exists(EXISTING_RADIOMICS_TEST_CSV):
#     print(f"WARNING: Existing selected test radiomics CSV not found at {EXISTING_RADIOMICS_TEST_CSV}")

os.makedirs(OUTPUT_COMBINED_DIR, exist_ok=True)

# Determine expected feature dimension
if MODEL_DEPTH in [10, 18, 34]:
    # EXPECTED_DEEP_FEAT_DIM = 512 * 2 # 1024
    EXPECTED_DEEP_FEAT_DIM = 256
else:
    EXPECTED_DEEP_FEAT_DIM = 512 * 4 * 2 # 4096
print(f"Configuration: Model Depth={MODEL_DEPTH}, Expected Feature Dim={EXPECTED_DEEP_FEAT_DIM}, Device={DEVICE}")
print(f"Output directory for combined files: {OUTPUT_COMBINED_DIR}")


# ==============================================================================
# == Dataset Definition (Modified for Extraction) ==
# ==============================================================================
# (Using the MRIDataset3DFusionForExtraction class from the previous response)
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
            g_data = nib.load(g_path).get_fdata(dtype=np.float32)
            g_data = np.expand_dims(np.transpose(g_data, (2,0,1)), axis=0)

            g_lbl_path = self.global_label_paths[idx]
            g_lbl_data = nib.load(g_lbl_path).get_fdata(dtype=np.float32)
            g_lbl_data = (g_lbl_data > 0).astype(np.float32)
            g_lbl_data = np.expand_dims(np.transpose(g_lbl_data, (2,0,1)), axis=0)

            l_path = self.local_image_paths[idx]
            l_data = nib.load(l_path).get_fdata(dtype=np.float32)
            l_data = np.expand_dims(np.transpose(l_data, (2,0,1)), axis=0)

            l_lbl_path = self.local_label_paths[idx]
            l_lbl_data = nib.load(l_lbl_path).get_fdata(dtype=np.float32)
            l_lbl_data = (l_lbl_data > 0).astype(np.float32)
            l_lbl_data = np.expand_dims(np.transpose(l_lbl_data, (2,0,1)), axis=0)
        except Exception as e:
             print(f"ERROR loading data for index {idx}, ID '{img_id_str}'. Error: {e}")
             # Return placeholder tensors and the ID string
             ph_g = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
             ph_g_lbl = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
             ph_l = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
             ph_l_lbl = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
             ph_g_2ch = torch.cat([ph_g, ph_g*ph_g_lbl], dim=0)
             ph_l_2ch = torch.cat([ph_l, ph_l*ph_l_lbl], dim=0)
             # Mark data as invalid by returning None for tensors? Or just return placeholders.
             # Let's return placeholders and the ID, the extraction function will handle missing features later.
             return ph_g_2ch, ph_l_2ch, img_id_str # Return ID even on error

        # ========== Construct Dict ==========
        data_dict = {
            "image_global": torch.tensor(g_data, dtype=torch.float32),
            "label_global": torch.tensor(g_lbl_data, dtype=torch.float32),
            "image_local":  torch.tensor(l_data,  dtype=torch.float32),
            "label_local":  torch.tensor(l_lbl_data, dtype=torch.float32)
        }

        # ========== Apply Deterministic Transforms ==========
        try:
            data_dict = self.crop_local_transform(data_dict)
            data_dict = self.resize_global(data_dict)
            data_dict = self.resize_local(data_dict)
        except Exception as e:
            print(f"ERROR during transforms for index {idx}, ID '{img_id_str}'. Error: {e}")
            # Return placeholder tensors and the ID string
            ph_g = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
            ph_g_lbl = torch.zeros((1, *self.final_shape_global), dtype=torch.float32)
            ph_l = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
            ph_l_lbl = torch.zeros((1, *self.final_shape_local), dtype=torch.float32)
            ph_g_2ch = torch.cat([ph_g, ph_g*ph_g_lbl], dim=0)
            ph_l_2ch = torch.cat([ph_l, ph_l*ph_l_lbl], dim=0)
            return ph_g_2ch, ph_l_2ch, img_id_str # Return ID even on error

        # ========== Assemble 2-Channel Inputs ==========
        g_img = data_dict["image_global"]
        g_lbl = data_dict["label_global"]
        g_2ch = torch.cat([g_img, g_img * g_lbl], dim=0)

        l_img = data_dict["image_local"]
        l_lbl = data_dict["label_local"]
        l_2ch = torch.cat([l_img, l_img * l_lbl], dim=0)

        # ========== Return Data and Original ID String ==========
        return g_2ch, l_2ch, img_id_str


# ==============================================================================
# == Hook Function and Feature Extraction Logic ==
# ==============================================================================

features_from_hook = []
def hook_fn(module, input_tensor, output_tensor):
    """Hook to capture the input tensor."""
    global features_from_hook
    tensor_to_process = None
    if isinstance(input_tensor, tuple) and len(input_tensor) > 0:
        tensor_to_process = input_tensor[0]
    elif torch.is_tensor(input_tensor):
         tensor_to_process = input_tensor

    if tensor_to_process is not None:
        features = tensor_to_process.detach().cpu()
        for i in range(features.shape[0]):
            features_from_hook.append(features[i])
    else:
        print(f"Warning: Hook received unexpected input format: {type(input_tensor)}")

# MODIFIED to return ID -> feature dict
def extract_deep_features_dict(model, dataloader, device, expected_feature_dim):
    """
    Extracts features using the registered hook and returns a dictionary
    mapping original ID string to the feature vector (as numpy array).

    Args:
        model: The loaded PyTorch model.
        dataloader: DataLoader providing (g_vol, l_vol, batch_ids_list).
        device: The device to run the model on ('cuda' or 'cpu').
        expected_feature_dim: The expected dimension of the feature vector.

    Returns:
        dict: {id_string: np.ndarray(feature_dim,), ...}
              Contains entries only for samples where feature extraction succeeded.
    """
    global features_from_hook
    features_from_hook = [] # Reset global list

    feature_dict = {} # Store ID -> feature mapping
    processed_ids = set() # Track processed IDs

    model.eval()
    hook_handle = None
    try:
        hook_handle = model.fusion_classifier[2].register_forward_hook(hook_fn)
        # hook_handle = model.fusion_classifier.register_forward_hook(hook_fn)
        # print(f"Registered forward hook on '{type(model.fusion_classifier).__name__}'.") # Reduce verbosity
    except AttributeError:
        print("ERROR: Could not find 'fusion_classifier' attribute in the model."); return {}
    except Exception as e:
         print(f"ERROR registering hook: {e}"); return {}

    # print(f"Starting feature extraction...") # Reduce verbosity
    batch_start_hook_idx = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting Features")):
        try:
            # DataLoader now returns a list of ID strings for the batch
            g_vol, l_vol, batch_ids_list = batch
            g_vol, l_vol = g_vol.to(device), l_vol.to(device)
        except Exception as e:
            print(f"Error unpacking/moving batch {batch_idx}: {e}. Skipping."); continue

        # Store current hook list length to identify features added by this batch
        # Note: This part was refined. We process hook list *after* forward pass.

        try:
            _ = model(g_vol, l_vol) # Run forward pass, hook captures features
        except Exception as e:
            print(f"Error during model forward pass for batch {batch_idx}: {e}. Skipping feature assignment.")
            # We don't know how many features were added to hook list, difficult to recover reliably. Best to skip batch.
            # It's crucial the model forward pass is robust.
            continue

        # --- Process features captured by the hook for THIS batch ---
        current_batch_size = len(batch_ids_list) # Use length of ID list
        batch_end_hook_idx = batch_start_hook_idx + current_batch_size

        if batch_end_hook_idx > len(features_from_hook):
            print(f"Warning: Hook list length ({len(features_from_hook)}) mismatch after batch {batch_idx}. "
                  f"Expected end index {batch_end_hook_idx}. Assigning features may be incorrect. Skipping.")
        else:
            batch_features = features_from_hook[batch_start_hook_idx : batch_end_hook_idx]

            if len(batch_features) != current_batch_size:
                print(f"Warning: Mismatch in batch {batch_idx}! Expected {current_batch_size} features, "
                      f"hook captured {len(batch_features)}. Skipping feature assignment.")
            else:
                # Assign features to the dictionary using the ID strings
                for i, id_str in enumerate(batch_ids_list):
                    if id_str not in processed_ids:
                        # Verify feature dimension before adding
                        if batch_features[i].shape[0] == expected_feature_dim:
                             feature_dict[id_str] = batch_features[i].numpy()
                             processed_ids.add(id_str)
                        else:
                             print(f"Warning: Incorrect feature dim ({batch_features[i].shape[0]}) for ID '{id_str}'. Expected {expected_feature_dim}. Skipping.")
                    else:
                        # This shouldn't happen with shuffle=False if IDs are unique
                        print(f"Warning: Duplicate ID '{id_str}' encountered in batch {batch_idx}.")

        # Update hook list pointer for the next batch
        batch_start_hook_idx = batch_end_hook_idx


    if hook_handle: hook_handle.remove()
    # print("Removed forward hook.") # Reduce verbosity
    print(f"Extraction finished. Features extracted for {len(feature_dict)} unique IDs.")

    return feature_dict


# ==============================================================================
# == Helper Function to Generate File Paths ==
# ==============================================================================
# (Using the generate_paths_from_ids function from the previous response)
def generate_paths_from_ids(ids, img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l):
    """Generates lists of file paths based on IDs, checking for existence."""
    g_img_paths, g_lbl_paths, l_img_paths, l_lbl_paths = [], [], [], []
    valid_ids = []
    missing_files_log = {}

    # print(f"Attempting to generate paths for {len(ids)} IDs...") # Reduce verbosity
    for img_id in ids:
        expected_filename = f"{img_id}.nii.gz" # Assumes .nii.gz, adjust if needed

        g_img_p = os.path.join(img_dir_g, expected_filename)
        g_lbl_p = os.path.join(lbl_dir_g, expected_filename)
        l_img_p = os.path.join(img_dir_l, expected_filename)
        l_lbl_p = os.path.join(lbl_dir_l, expected_filename)

        required_paths = [g_img_p, g_lbl_p, l_img_p, l_lbl_p]
        current_missing = [p for p in required_paths if not os.path.exists(p)]

        if not current_missing:
            g_img_paths.append(g_img_p)
            g_lbl_paths.append(g_lbl_p)
            l_img_paths.append(l_img_p)
            l_lbl_paths.append(l_lbl_p)
            valid_ids.append(img_id) # Add ID only if all files exist
        else:
            missing_files_log[img_id] = current_missing

    num_missing_ids = len(missing_files_log)
    if num_missing_ids > 0:
        print(f"WARNING: Could not find all required NIfTI files for {num_missing_ids} out of {len(ids)} IDs.")
        count = 0
        for missing_id, missing_list in missing_files_log.items():
            if count < 3: print(f"  - ID '{missing_id}': Missing {missing_list}")
            count += 1
        if num_missing_ids > 3: print(f"  ... and {num_missing_ids - 3} more IDs with missing files.")
        print(f"--> Proceeding with {len(valid_ids)} IDs for which all files were found.")
    # else: # Reduce verbosity
        # print(f"Successfully found all required files for all {len(ids)} IDs.")

    return g_img_paths, g_lbl_paths, l_img_paths, l_lbl_paths, valid_ids


# ==============================================================================
# == Main Execution Block ==
# ==============================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 1. Load Model
    print("\n--- Loading Pretrained Model ---")
    model = BinaryResNet3D_FusionModel(model_depth=MODEL_DEPTH, freeze_branches=False)
    try:
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        state_dict_key = next((k for k in ['state_dict', 'model_state_dict'] if k in checkpoint), None)
        state_dict = checkpoint[state_dict_key] if state_dict_key else checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {PRETRAINED_MODEL_PATH}")
    except Exception as e:
        print(f"ERROR loading model weights: {e}. Exiting."); exit()
    model.to(DEVICE)

    # --- Process Function (to avoid code duplication) ---
    def process_dataset(csv_path, img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l, output_filename):
        print(f"\n--- Processing: {os.path.basename(csv_path)} ---")
        if not os.path.exists(csv_path):
            print(f"ERROR: Radiomics file not found: {csv_path}. Skipping.")
            return

        # Load existing radiomics data
        try:
            radiomics_df = pd.read_csv(csv_path)
            # Ensure ID column is string type for reliable merging
            if 'ID' not in radiomics_df.columns:
                 print(f"ERROR: 'ID' column not found in {csv_path}. Skipping.")
                 return
            radiomics_df['ID'] = radiomics_df['ID'].astype(str)
            original_ids = radiomics_df['ID'].tolist()
            print(f"Loaded {len(radiomics_df)} samples from {csv_path}.")
        except Exception as e:
            print(f"Error loading radiomics CSV {csv_path}: {e}. Skipping.")
            return

        # Generate NIfTI paths based on IDs from the radiomics CSV
        g_img_paths, g_lbl_paths, l_img_paths, l_lbl_paths, valid_ids_for_nifti = generate_paths_from_ids(
            original_ids, img_dir_g, lbl_dir_g, img_dir_l, lbl_dir_l
        )

        if not valid_ids_for_nifti:
            print("ERROR: No valid NIfTI files found for the IDs in the CSV. Cannot extract deep features.")
            # Optionally save the radiomics data alone if needed, but merging isn't possible.
            # output_path = os.path.join(OUTPUT_COMBINED_DIR, output_filename)
            # radiomics_df.to_csv(output_path, index=False)
            # print(f"Saved original radiomics data (only) to: {output_path}")
            return

        # Create Dataset and DataLoader for VALID IDs only
        dataset = MRIDataset3DFusionForExtraction(
            original_ids=valid_ids_for_nifti, # Pass only the valid IDs
            global_image_paths=g_img_paths, global_label_paths=g_lbl_paths,
            local_image_paths=l_img_paths, local_label_paths=l_lbl_paths,
            is_center3=IS_CENTER3
        )
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        # Extract deep features -> returns dict {id_str: feature_np_array}
        deep_feature_dict = extract_deep_features_dict(model, dataloader, DEVICE, EXPECTED_DEEP_FEAT_DIM)

        if not deep_feature_dict:
            print("ERROR: Deep feature extraction failed or returned empty.")
             # Save original radiomics only
            # output_path = os.path.join(OUTPUT_COMBINED_DIR, output_filename)
            # radiomics_df.to_csv(output_path, index=False)
            # print(f"Saved original radiomics data (only) to: {output_path}")
            return

        # --- Combine Features using Pandas ---
        print("Combining radiomics and deep features...")
        # Convert deep feature dict to DataFrame
        deep_df = pd.DataFrame.from_dict(deep_feature_dict, orient='index')
        # Create column names for deep features
        deep_df.columns = [f'deep_feat_{i}' for i in range(EXPECTED_DEEP_FEAT_DIM)]
        deep_df.index.name = 'ID' # Set the index name to 'ID'
        deep_df = deep_df.reset_index() # Make 'ID' a regular column for merging

        # Merge with radiomics data based on 'ID'
        # Use 'inner' join to keep only IDs present in BOTH radiomics AND successfully extracted deep features
        combined_df = pd.merge(radiomics_df, deep_df, on='ID', how='inner')
        print(f"Combined DataFrame shape: {combined_df.shape}")
        print(f" - Kept {len(combined_df)} samples present in both radiomics and successful deep feature extraction.")
        print(f" - Lost {len(radiomics_df) - len(combined_df)} samples from original radiomics due to merge.")

        # --- Save Combined Data ---
        output_path = os.path.join(OUTPUT_COMBINED_DIR, output_filename)
        try:
            combined_df.to_csv(output_path, index=False)
            print(f"Successfully saved combined features to: {output_path}")
        except Exception as e:
            print(f"Error saving combined CSV to {output_path}: {e}")

    # --- Execute Processing ---
    process_dataset(
        EXISTING_RADIOMICS_TRAIN_CSV,
        GLOBAL_IMG_TR_DIR, GLOBAL_LBL_TR_DIR, LOCAL_IMG_TR_DIR, LOCAL_LBL_TR_DIR,
        "train_features_RD.csv" # Output filename for train
    )

    # Check if test directories and CSV exist before processing test set
    test_dirs_exist = all(os.path.isdir(d) for d in [GLOBAL_IMG_TS_DIR, GLOBAL_LBL_TS_DIR, LOCAL_IMG_TS_DIR, LOCAL_LBL_TS_DIR])
    if os.path.exists(EXISTING_RADIOMICS_TEST_CSV) and test_dirs_exist:
         process_dataset(
             EXISTING_RADIOMICS_TEST_CSV,
             GLOBAL_IMG_TS_DIR, GLOBAL_LBL_TS_DIR, LOCAL_IMG_TS_DIR, LOCAL_LBL_TS_DIR,
             "test_features_RD.csv" # Output filename for test
         )
    elif not os.path.exists(EXISTING_RADIOMICS_TEST_CSV):
        print(f"\nWARNING: Existing selected test radiomics CSV not found at {EXISTING_RADIOMICS_TEST_CSV}. Skipping test set processing.")
    else: # CSV exists but dirs don't
        print(f"\nWARNING: Testing NIfTI data directories not found ({GLOBAL_IMG_TS_DIR}, etc.). Skipping test set processing.")


    print("\n--- Feature extraction and combination script finished ---")