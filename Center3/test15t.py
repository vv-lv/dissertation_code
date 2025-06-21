import numpy as np
from scipy.ndimage import binary_fill_holes
import nibabel as nib
# from skimage.transform import resize # Not used if using MONAI resample
# from scipy.ndimage import map_coordinates # Not used if using MONAI resample
import os
from tqdm import tqdm
from monai.transforms import (
    NormalizeIntensity,
    Spacing
)
from monai.data import MetaTensor
# import torch # Not directly needed unless using torch tensors outside MONAI

# ---- Required Import ----
from skimage.filters import threshold_otsu
# -------------------------


def modify_affine_no_rotation(
    old_affine: np.ndarray,
    old_spacing: tuple,  # (Sx, Sy, Sz) - Use absolute values
    new_spacing: tuple,  # (Sx_new, Sy_new, Sz_new) - Use absolute values
    bbox: list = None    # [ [x0,x1], [y0,y1], [z0,z1] ] voxel indices
):
    """
    Updates affine based on "no rotation/skew" assumption:
      1) Calculates translation offset based on bbox (if provided).
      2) Updates the diagonal elements of the affine matrix to reflect new_spacing.
    """
    # Ensure input affine is float64 for precision
    new_affine = old_affine.copy().astype(np.float64)
    abs_old_spacing = np.abs(np.array(old_spacing))
    abs_new_spacing = np.abs(np.array(new_spacing))

    # === 1) Apply translation shift if bbox is provided ===
    if bbox is not None:
        x0, y0, z0 = bbox[0][0], bbox[1][0], bbox[2][0] # Min coords of bbox
        origin_offset = new_affine[:3, :3] @ np.array([x0, y0, z0])
        new_affine[:3, 3] += origin_offset
        # print(f"  [modify_affine] Applied bbox shift for origin ({x0},{y0},{z0}). Offset: {origin_offset}")


    # === 2) Update diagonal elements for new spacing ===
    valid_indices = np.where(abs_old_spacing > 1e-6)[0]
    for i in valid_indices:
        original_sign = np.sign(new_affine[i, i]) if new_affine[i, i] != 0 else 1
        new_affine[i, i] = original_sign * abs_new_spacing[i]
        # print(f"  [modify_affine] Updated spacing for axis {i} to {new_affine[i, i]:.4f}")

    return new_affine


# 步骤 1: 生成前景区域模板 (foreground_mask) using Otsu - (No change from previous Otsu version)
def generate_nonzero_mask(data):
    """
    Generates a foreground mask based on Otsu's intensity thresholding.
    Handles 3D (H, W, D) or 4D (C, H, W, D) data.
    For 4D, computes mask for each channel and combines with logical OR.
    Fills holes in the resulting mask.
    """
    if data.ndim == 4: # Assumes (C, H, W, D)
        print("  [generate_nonzero_mask] Input is 4D. Calculating mask per channel and combining.")
        combined_mask = np.zeros(data.shape[1:], dtype=bool) # Shape (H, W, D)
        num_channels = data.shape[0]
        for c in range(num_channels):
            channel_data = data[c]
            if np.ptp(channel_data) == 0: # Peak-to-peak is zero (constant)
                 print(f"    Channel {c}: Data is constant. Using non-zero check as fallback.")
                 channel_mask = channel_data != 0
            else:
                try:
                    thresh = threshold_otsu(channel_data)
                    channel_mask = channel_data > thresh
                    print(f"    Channel {c}: Otsu thresh={thresh:.4f}, Mask sum={np.sum(channel_mask)}")
                except ValueError as e:
                    print(f"    Channel {c}: Otsu failed ({e}). Using non-zero fallback for this channel.")
                    channel_mask = channel_data != 0
            combined_mask = combined_mask | channel_mask
        final_mask = combined_mask
    elif data.ndim == 3: # Assumes (H, W, D)
        print("  [generate_nonzero_mask] Input is 3D. Calculating mask.")
        if np.ptp(data) == 0:
            print("  [generate_nonzero_mask] Data is constant. Using non-zero check as fallback.")
            final_mask = data != 0
        else:
            try:
                thresh = threshold_otsu(data)
                final_mask = data > thresh
                print(f"  [generate_nonzero_mask] Otsu threshold: {thresh:.4f}")
            except ValueError as e:
                print(f"  [generate_nonzero_mask] Otsu thresholding failed ({e}). Falling back to non-zero check.")
                final_mask = data != 0
    else:
            raise ValueError(f"Unsupported data dimension for mask generation: {data.ndim}")

    print("  [generate_nonzero_mask] Filling holes in the final mask...")
    final_mask = binary_fill_holes(final_mask.astype(np.int8)).astype(bool)

    foreground_voxel_count = np.sum(final_mask)
    print(f"  [generate_nonzero_mask] Final mask generated. Foreground voxel count: {foreground_voxel_count}")
    if foreground_voxel_count == 0:
        print("  [generate_nonzero_mask] WARNING: Foreground mask is empty after Otsu and hole filling!")

    return final_mask


# 步骤 2: 计算裁剪区域的边界框 (bounding_box) - (No change)
def get_bounding_box(nonzero_mask):
    """
    nonzero_mask.shape = (X, Y, Z) or (H, W, D) depending on convention
    Returns a bbox = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    coords = np.where(nonzero_mask) # Find indices of True elements
    if not coords[0].size: # Handle empty mask
        print("  [get_bounding_box] WARNING: Non-zero mask is empty. Returning full image bounds.")
        return [[0, nonzero_mask.shape[0]], [0, nonzero_mask.shape[1]], [0, nonzero_mask.shape[2]]]

    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    return [[x_min, x_max + 1], [y_min, y_max + 1], [z_min, z_max + 1]]


# 步骤 3: 根据bounding box裁剪图像 - (No change)
def crop_image(data, bbox):
    """
    Crops the image data based on the provided bounding box.
    Handles 3D (H, W, D) or 4D (C, H, W, D) data.
    bbox = [[h0,h1], [w0,w1], [d0,d1]]
    """
    h0, h1 = bbox[0]
    w0, w1 = bbox[1]
    d0, d1 = bbox[2]

    if data.ndim == 3:
        cropped = data[h0:h1, w0:w1, d0:d1]
    elif data.ndim == 4:
        cropped = data[:, h0:h1, w0:w1, d0:d1]
    else:
         raise ValueError(f"Unsupported data dimension for cropping: {data.ndim}")
    print(f"  [crop_image] Cropped data shape: {cropped.shape}")
    return cropped


# 步骤 4: 裁剪标签图像 - (No change)
def crop_label(seg, bbox):
    """
    Crops the label/segmentation data based on the bounding box.
    Handles 3D (H, W, D) or 4D (C, H, W, D) data.
    bbox = [[h0,h1], [w0,w1], [d0,d1]]
    """
    h0, h1 = bbox[0]
    w0, w1 = bbox[1]
    d0, d1 = bbox[2]

    if seg.ndim == 3:
        cropped_seg = seg[h0:h1, w0:w1, d0:d1]
    elif seg.ndim == 4:
        cropped_seg = seg[:, h0:h1, w0:w1, d0:d1]
    else:
         raise ValueError(f"Unsupported label dimension for cropping: {seg.ndim}")
    print(f"  [crop_label] Cropped label shape: {cropped_seg.shape}")
    return cropped_seg


# --- REMOVED should_use_nonzero_mask function ---


# 主程序：裁剪图像数据和标签 - MODIFIED return values
def preprocess_data(data, seg):
    """Performs cropping based on intensity mask."""
    print("[preprocess_data] Starting...")
    # original_shape = data.shape[-3:] # Not needed anymore

    foreground_mask = generate_nonzero_mask(data)
    bbox = get_bounding_box(foreground_mask)
    print(f"  [preprocess_data] Calculated BBox: {bbox}")

    cropped_data = crop_image(data, bbox)
    cropped_seg = crop_label(seg, bbox)

    # cropped_shape = cropped_data.shape[-3:] # Not needed anymore
    # use_nonzero_mask_for_norm = should_use_nonzero_mask(original_shape, cropped_shape) # REMOVED

    print("[preprocess_data] Finished.")
    # Return only cropped data, label, and bbox
    return cropped_data, cropped_seg, bbox


# 计算目标spacing - SIMPLIFIED (removed anisotropy logic)
def calculate_target_spacing(spacings, shapes, target_spacing_percentile=50, anisotropy_threshold=3):
    """Calculates the target spacing, typically the median over the dataset."""
    print("[calculate_target_spacing] Calculating target spacing...")
    spacings_arr = np.vstack(spacings)
    # shapes_arr = np.vstack(shapes) # Shape info no longer needed for this simplified version

    # Calculate median (or other percentile) spacing
    target = np.percentile(spacings_arr, target_spacing_percentile, axis=0)
    print(f"  Median spacing across dataset (used as target): {target}")

    # --- REMOVED Anisotropy Check Logic ---
    # The complex logic to check for spacing/voxel anisotropy and adjust
    # target spacing for specific axes (do_separate_z) has been removed
    # as requested. We simply use the calculated percentile spacing.

    print(f"  Final target spacing: {target}")
    print("[calculate_target_spacing] Finished.")
    # Return target spacing, and placeholder False/None for the removed logic
    return target, False, None


# 计算新的图像尺寸 - (No change needed, but check usage if anisotropy logic removed)
# This function is actually NOT directly used in the MONAI resampling workflow,
# as MONAI calculates the output shape internally based on spacing.
# It might be useful for debugging or alternative resampling methods.
# def calculate_new_shape(original_spacing, target_spacing, shape):
#     """Calculates the theoretical new shape after resampling."""
#     original_spacing = np.array(original_spacing)
#     target_spacing = np.array(target_spacing)
#     shape = np.array(shape)
#     valid_idx = np.where(target_spacing > 1e-6)[0]
#     new_shape = np.array(shape) # Initialize with old shape
#     if valid_idx.size > 0:
#          new_shape[valid_idx] = np.round(((original_spacing[valid_idx] / target_spacing[valid_idx]) * shape[valid_idx])).astype(int)
#     # print(f"  [calculate_new_shape] Original shape: {shape}, Original spacing: {original_spacing}, Target spacing: {target_spacing} -> New shape: {new_shape}")
#     return new_shape


# MONAI Resampling Function - MODIFIED affine handling and removed dtype from label Spacing
def monai_resample_image_and_label(image_3d, label_3d, old_affine, target_spacing):
    """
    Resamples image and label using MONAI Spacing transform.
    Ensures affine is float64 for transformations.
    Input arrays should be (H, W, D).
    Input affine is the affine of the input arrays BEFORE resampling.
    """
    print("[monai_resample_image_and_label] Starting MONAI resampling...")
    print(f"  Input image shape: {image_3d.shape}, Input label shape: {label_3d.shape}")
    print(f"  Target spacing: {target_spacing}")

    # *** FIX: Ensure affine is float64 before passing to MetaTensor ***
    affine_float64 = old_affine.astype(np.float64)
    print(f"  Input affine (ensured float64):\n{affine_float64}")

    # Add channel dim: (H, W, D) -> (1, H, W, D)
    image_4d = np.expand_dims(image_3d, axis=0)
    label_4d = np.expand_dims(label_3d, axis=0) # Label should be uint8 or similar int type here

    # Create MetaTensors with float64 affine
    image_mt = MetaTensor(image_4d, affine=affine_float64)
    # Label data itself is int, but affine must be float
    label_mt = MetaTensor(label_4d, affine=affine_float64)

    # Define Spacing transforms
    spacing_image = Spacing(pixdim=target_spacing, mode="bilinear", align_corners=True, dtype=np.float32, recompute_affine=True)
    # *** FIX: Removed dtype=np.uint8 from label spacing ***
    # Let MONAI handle output type based on input MetaTensor or cast later
    spacing_label = Spacing(pixdim=target_spacing, mode="nearest", align_corners=True, recompute_affine=True)

    # Apply transforms
    new_image_t = spacing_image(image_mt)
    new_label_t = spacing_label(label_mt)

    # Get the new affine (should be float)
    new_affine = new_image_t.affine.cpu().numpy()
    print(f"  Output affine from MONAI:\n{new_affine}")

    # Convert back to numpy arrays and remove channel dim
    new_image_3d = new_image_t.squeeze(0).cpu().numpy()
    new_label_3d = new_label_t.squeeze(0).cpu().numpy()

    # MONAI's nearest neighbor might output float, explicitly cast label back to int
    new_label_3d = np.round(new_label_3d).astype(np.uint8)

    print(f"  Output image shape: {new_image_3d.shape}, Output label shape: {new_label_3d.shape} (dtype: {new_label_3d.dtype})")
    print("[monai_resample_image_and_label] Finished.")
    return new_image_3d, new_label_3d, new_affine

# --- Save Functions - NO CHANGE NEEDED ---
def save_processed_image(output_dir, image_data, affine, subdir, filename):
    output_path = os.path.join(output_dir, subdir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_img = nib.Nifti1Image(image_data.astype(np.float32), affine)
    nib.save(processed_img, output_path)

def save_processed_segmentation(output_dir, seg_data, affine, subdir, filename):
    output_path = os.path.join(output_dir, subdir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_seg = nib.Nifti1Image(seg_data.astype(np.uint8), affine) # Ensure uint8 for label saving
    nib.save(processed_seg, output_path)

# --- Load Function - NO CHANGE NEEDED ---
def load_nifti_get_array_and_affine(path):
    nif = nib.load(path)
    # Load data, ensure it's float for intensity processing initially
    data = nif.get_fdata(dtype=np.float32)
    affine = nif.affine
    # Get spacing using header zooms for reliability
    spacing = nif.header.get_zooms()[:3]
    return data, affine, tuple(spacing)


# --- Main Resampling Loop - MODIFIED ---
def resample_dataset(data_dir, output_dir, target_spacing_percentile=50, anisotropy_threshold=3): # anisotropy_threshold now unused here but kept for signature consistency
    print(f"Starting dataset resampling from '{data_dir}' to '{output_dir}'...")

    # os.makedirs(os.path.join(output_dir, 'imagesTr'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'labelsTr'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'imagesTs_3'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labelsTs_3'), exist_ok=True)

    spacings = []
    shapes = []
    imageTr_dir = os.path.join(data_dir, 'imagesTr')
    imageTs_dir = os.path.join(data_dir, 'imagesTs_3')
    imageTr_paths = sorted([os.path.join(imageTr_dir, fname) for fname in os.listdir(imageTr_dir) if fname.endswith(('.nii', '.nii.gz'))])
    imageTs_paths = sorted([os.path.join(imageTs_dir, fname) for fname in os.listdir(imageTs_dir) if fname.endswith(('.nii', '.nii.gz'))])

    print("Gathering spacing and shape information from Training set...")
    for img_path in tqdm(imageTr_paths, desc="Analyzing Tr images"):
        try:
            img = nib.load(img_path)
            spacing = img.header.get_zooms()[:3]
            spacings.append(spacing)
            shapes.append(img.shape[:3]) # Shape info not directly used anymore, but maybe useful later
        except Exception as e:
            print(f"Warning: Could not load or get info from {img_path}. Skipping. Error: {e}")

    if not spacings:
         raise ValueError("No valid images found in imagesTr to calculate target spacing.")

    # Calculate target spacing (simplified version)
    # The returned do_separate_z and axis are ignored (placeholders _, _)
    target_spacing, _, _ = calculate_target_spacing(spacings, shapes, target_spacing_percentile)

    # print("\nProcessing Training Images (imagesTr / labelsTr)...")
    # for img_path in tqdm(imageTr_paths, desc="Processing Tr"):
    #     filename = os.path.basename(img_path)
    #     print(f"\n--- Processing {filename} ---")
    #     try:
    #         # 1) Load Image + Label + Spacing
    #         img_data, img_affine, img_spacing = load_nifti_get_array_and_affine(img_path)
    #         seg_path = img_path.replace(imageTr_dir, os.path.join(data_dir, 'labelsTr'))
    #         if not os.path.exists(seg_path):
    #              print(f"  Warning: Label file not found for {filename} at {seg_path}. Skipping.")
    #              continue
    #         seg_data, seg_affine, _ = load_nifti_get_array_and_affine(seg_path)
    #         # Ensure label data is integer type (e.g., uint8) after loading
    #         seg_data = seg_data.astype(np.uint8)
    #         if not np.allclose(img_affine, seg_affine, atol=1e-3):
    #              print(f"  Warning: Affine mismatch between image and label for {filename}. Using image affine.")

    #         print(f"  Original shape: {img_data.shape}, Original spacing: {img_spacing}")

    #         # 2) Crop using intensity-based mask
    #         # Unpack return values (use_nonzero_mask_for_norm removed)
    #         cropped_img, cropped_seg, bbox = preprocess_data(img_data, seg_data)

    #         # 3) Calculate Affine for the *Cropped* Data BEFORE resampling
    #         print("  Calculating affine for cropped data...")
    #         affine_of_cropped_data = modify_affine_no_rotation(
    #             old_affine=img_affine, old_spacing=img_spacing,
    #             new_spacing=img_spacing, bbox=bbox
    #         )
    #         print(f"  Affine after bbox correction:\n{affine_of_cropped_data}")

    #         # 4) Resample using MONAI
    #         resampled_img, resampled_seg, final_affine = monai_resample_image_and_label(
    #             cropped_img, cropped_seg,
    #             old_affine=affine_of_cropped_data,
    #             target_spacing=target_spacing
    #         )

    #         # 5) Normalize the *resampled* image (Global normalization)
    #         print("  Normalizing resampled image...")
    #         resampled_img_4d = np.expand_dims(resampled_img, axis=0)
    #         # Normalize globally (based on all voxels)
    #         transform = NormalizeIntensity(nonzero=False, channel_wise=False)
    #         normalized_img_4d = transform(resampled_img_4d)
    #         normalized_img_3d = normalized_img_4d.squeeze(0)

    #         # 6) Save Processed Image and Label
    #         print("  Saving processed files...")
    #         save_processed_image(output_dir, normalized_img_3d, final_affine, 'imagesTr', filename)
    #         # Label should already be uint8 from monai_resample or explicit cast
    #         save_processed_segmentation(output_dir, resampled_seg, final_affine, 'labelsTr', filename)
    #         print(f"  Finished saving {filename}. Final shape: {normalized_img_3d.shape}")

    #     except Exception as e:
    #         print(f"\n****** ERROR processing {filename}: {e} ******")
    #         import traceback
    #         traceback.print_exc()
    #         print(f"****** Skipping {filename} due to error ******")


    print("\nProcessing Test Images (imagesTs / labelsTs)...")
    # Apply the *same* target spacing calculated from the training set
    for img_path in tqdm(imageTs_paths, desc="Processing Ts"):
        filename = os.path.basename(img_path)
        print(f"\n--- Processing {filename} ---")
        try:
            # 1) Load Image + Label + Spacing
            img_data, img_affine, img_spacing = load_nifti_get_array_and_affine(img_path)
            seg_path = img_path.replace(imageTs_dir, os.path.join(data_dir, 'labelsTs_3'))
            if not os.path.exists(seg_path):
                 print(f"  Warning: Label file not found for {filename} at {seg_path}. Skipping.")
                 continue
            seg_data, seg_affine, _ = load_nifti_get_array_and_affine(seg_path)
            seg_data = seg_data.astype(np.uint8) # Ensure label is int
            if not np.allclose(img_affine, seg_affine, atol=1e-3):
                 print(f"  Warning: Affine mismatch between image and label for {filename}. Using image affine.")

            print(f"  Original shape: {img_data.shape}, Original spacing: {img_spacing}")

            # 2) Crop using intensity-based mask
            cropped_img, cropped_seg, bbox = preprocess_data(img_data, seg_data)

            # 3) Calculate Affine for the *Cropped* Data BEFORE resampling
            print("  Calculating affine for cropped data...")
            affine_of_cropped_data = modify_affine_no_rotation(
                old_affine=img_affine, old_spacing=img_spacing,
                new_spacing=img_spacing, bbox=bbox
            )
            print(f"  Affine after bbox correction:\n{affine_of_cropped_data}")

            # 4) Resample using MONAI
            resampled_img, resampled_seg, final_affine = monai_resample_image_and_label(
                cropped_img, cropped_seg,
                old_affine=affine_of_cropped_data,
                target_spacing=target_spacing # Use target spacing from Tr set
            )

            # 5) Normalize the *resampled* image (Global normalization)
            print("  Normalizing resampled image...")
            resampled_img_4d = np.expand_dims(resampled_img, axis=0)
            transform = NormalizeIntensity(nonzero=False, channel_wise=False)
            normalized_img_4d = transform(resampled_img_4d)
            normalized_img_3d = normalized_img_4d.squeeze(0)

            # 6) Save Processed Image and Label
            print("  Saving processed files...")
            save_processed_image(output_dir, normalized_img_3d, final_affine, 'imagesTs_3', filename)
            save_processed_segmentation(output_dir, resampled_seg, final_affine, 'labelsTs_3', filename)
            print(f"  Finished saving {filename}. Final shape: {normalized_img_3d.shape}")

        except Exception as e:
            print(f"\n****** ERROR processing {filename}: {e} ******")
            import traceback
            traceback.print_exc()
            print(f"****** Skipping {filename} due to error ******")

    print("\nDataset resampling finished.")


# --- Example Run ---
if __name__ == "__main__":
    # Set your input and output directories
    data_dir = "/home/vipuser/Desktop/Data/Task02_PASp61"  # <--- CHECK THIS PATH
    output_dir = "/home/vipuser/Desktop/Data/Task02_PASp62" # <--- CHECK THIS PATH

    if not os.path.isdir(data_dir):
        print(f"ERROR: Input data directory not found: {data_dir}")
    elif not os.path.isdir(os.path.join(data_dir, 'imagesTr')):
         print(f"ERROR: Input 'imagesTr' subdirectory not found in {data_dir}")
    else:
        resample_dataset(data_dir, output_dir)