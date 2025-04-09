"""
feature_extraction.py

Module for extracting radiomic features from a 3D multi-labeled lesion mask.
This version computes features for each lesion (each unique nonzero label in the mask)
using the entire 3D lesion for intensity and geometric features, and averaging 2D texture
features (GLCM and GLRLM) computed slice-by-slice where the lesion is present.
"""

import nibabel as nib
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def quantize_image(image, levels=256):
    """
    Normalize and quantize a 2D image to integer levels.
    
    Parameters
    ----------
    image : np.ndarray
        The input 2D image.
    levels : int
        Number of gray levels.
    
    Returns
    -------
    np.ndarray
        Quantized image as uint8.
    """
    min_val, max_val = np.min(image), np.max(image)
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)
    image_norm = (image - min_val) / (max_val - min_val)
    image_quant = (image_norm * (levels - 1)).astype(np.uint8)
    return image_quant

def compute_first_order_features(intensities, modality="T1", bins=256):
    """
    Compute additional first order features from an array of intensity values.

    Parameters
    ----------
    intensities : np.ndarray
        1D array of intensity values from the lesion.
    modality : str, optional
        String identifier for the modality (e.g., "T1", "T2", "QSM").
    bins : int, optional
        Number of bins to use for the entropy calculation (default is 256).

    Returns
    -------
    dict
        Dictionary with computed first order features.
    """
    features = {}
    prefix = modality + "_"  # Create a prefix for each feature name

    # Compute basic statistics
    features[prefix + "min"] = float(np.min(intensities))                  # Minimum intensity
    features[prefix + "max"] = float(np.max(intensities))                  # Maximum intensity
    features[prefix + "range"] = features[prefix + "max"] - features[prefix + "min"]  # Intensity range
    features[prefix + "median"] = float(np.median(intensities))            # Median intensity
    features[prefix + "percentile_10"] = float(np.percentile(intensities, 10))  # 10th percentile
    features[prefix + "percentile_90"] = float(np.percentile(intensities, 90))  # 90th percentile

    # Compute shape of the distribution
    features[prefix + "skewness"] = float(skew(intensities))               # Measure of asymmetry
    features[prefix + "kurtosis"] = float(kurtosis(intensities))             # Measure of tail weight

    # Compute energy: sum of squared intensities (captures overall magnitude)
    features[prefix + "energy"] = float(np.sum(np.square(intensities)))

    # Compute entropy: measure of randomness
    # First, build a normalized histogram of intensity values.
    hist, _ = np.histogram(intensities, bins=bins, density=True)
    # Filter out zero probabilities to avoid log(0) issues.
    hist = hist[hist > 0]
    features[prefix + "entropy"] = float(-np.sum(hist * np.log(hist)))

    return features


def compute_glcm_features_2d(image, distances=[1], angles=None, levels=256):
    """
    Compute GLCM properties from a 2D quantized image.
    
    Parameters
    ----------
    image : np.ndarray
        A quantized 2D image (dtype uint8).
    distances : list
        List of pixel pair distance offsets.
    angles : list, optional
        List of angles in radians. If None, defaults to [0, π/4, π/2, 3π/4].
    levels : int
        Number of gray levels.
    
    Returns
    -------
    dict
        Dictionary with averaged GLCM features over all distances and angles.
    """

    # Set default angles if not provided (0°, 45°, 90°, 135° in radians)
    if angles is None:
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Compute the GLCM matrix over multiple distances and angles.
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # Compute base properties using graycoprops and average them across distances and angles.
    base_features = {
        "glcm_contrast": float(np.mean(graycoprops(glcm, 'contrast'))),
        "glcm_dissimilarity": float(np.mean(graycoprops(glcm, 'dissimilarity'))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, 'homogeneity'))),
        "glcm_energy": float(np.mean(graycoprops(glcm, 'energy'))),
        "glcm_correlation": float(np.mean(graycoprops(glcm, 'correlation'))),
        "glcm_asm": float(np.mean(graycoprops(glcm, 'ASM')))
    }

    # Average the GLCM matrix over all distances and angles to create a representative matrix.
    P_avg = np.mean(glcm, axis=(2, 3))

    # Small epsilon to avoid log(0)
    eps = 1e-10

    # Additional features computed on the averaged matrix.
    glcm_entropy = -np.sum(P_avg * np.log(P_avg + eps))
    glcm_max_probability = float(np.max(P_avg))

    # Create index matrices for computing higher-order moments.
    i = np.arange(P_avg.shape[0])[:, np.newaxis]
    j = np.arange(P_avg.shape[1])[np.newaxis, :]

    # Compute marginal means.
    mu_i = np.sum(i * P_avg)
    mu_j = np.sum(j * P_avg)

    # Compute cluster shade and cluster prominence.
    cluster_shade = np.sum(((i + j - mu_i - mu_j) ** 3) * P_avg)
    cluster_prominence = np.sum(((i + j - mu_i - mu_j) ** 4) * P_avg)

    additional_features = {
        "glcm_entropy": float(glcm_entropy),
        "glcm_max_probability": glcm_max_probability,
        "glcm_cluster_shade": float(cluster_shade),
        "glcm_cluster_prominence": float(cluster_prominence)
    }

    # Combine and return all features.
    all_features = {**base_features, **additional_features}
    return all_features



def compute_glrlm_features_2d(image, levels=256):
    """
    Compute GLRLM features from a 2D quantized image.
    
    Parameters
    ----------
    image : np.ndarray
        A quantized 2D image (dtype uint8).
    levels : int, optional
        Number of gray levels (default is 256).
    
    Returns
    -------
    dict
        Dictionary with GLRLM features:
         - glrlm_mean_run: Mean run length.
         - glrlm_short_run_emphasis: Short-run emphasis (SRE).
         - glrlm_long_run_emphasis: Long-run emphasis (LRE).
         - glrlm_gray_level_nonuniformity: Gray-level nonuniformity (GLN).
         - glrlm_run_length_nonuniformity: Run-length nonuniformity (RLN).
    """

    # Collect runs as tuples: (gray level, run length)
    runs = []
    for row in image:
        if len(row) == 0:
            continue
        current_val = row[0]
        run_length = 1
        for pixel in row[1:]:
            if pixel == current_val:
                run_length += 1
            else:
                runs.append((current_val, run_length))
                current_val = pixel
                run_length = 1
        runs.append((current_val, run_length))
    
    if not runs:
        return {
            "glrlm_mean_run": 0.0,
            "glrlm_short_run_emphasis": 0.0,
            "glrlm_long_run_emphasis": 0.0,
            "glrlm_gray_level_nonuniformity": 0.0,
            "glrlm_run_length_nonuniformity": 0.0
        }
    
    # Determine the maximum run length observed.
    max_run_length = max(run[1] for run in runs)
    
    # Build the GLRLM matrix with shape (levels, max_run_length).
    # Each element [i, j] counts the number of runs for gray level i with run length (j+1).
    glrlm = np.zeros((levels, max_run_length), dtype=np.float64)
    for (g, r) in runs:
        glrlm[int(g), r-1] += 1  # run length index is r-1 because run lengths start at 1
    
    # Total number of runs
    N_runs = np.sum(glrlm)
    if N_runs == 0:
        return {
            "glrlm_mean_run": 0.0,
            "glrlm_short_run_emphasis": 0.0,
            "glrlm_long_run_emphasis": 0.0,
            "glrlm_gray_level_nonuniformity": 0.0,
            "glrlm_run_length_nonuniformity": 0.0
        }
    
    # Create an array of run lengths (1-indexed)
    run_lengths = np.arange(1, max_run_length + 1, dtype=np.float64)
    
    # Compute mean run length: weighted average of run lengths
    mean_run = np.sum(glrlm * run_lengths) / N_runs
    
    # Short Run Emphasis (SRE): emphasizes short runs
    sre = np.sum(glrlm / (run_lengths**2)) / N_runs
    
    # Long Run Emphasis (LRE): emphasizes long runs
    lre = np.sum(glrlm * (run_lengths**2)) / N_runs
    
    # Gray-Level Nonuniformity (GLN): measures the variability of gray levels
    row_sums = np.sum(glrlm, axis=1)
    gln = np.sum(row_sums**2) / N_runs
    
    # Run-Length Nonuniformity (RLN): measures the variability of run lengths
    col_sums = np.sum(glrlm, axis=0)
    rln = np.sum(col_sums**2) / N_runs
    
    return {
        "glrlm_mean_run": float(mean_run),
        "glrlm_short_run_emphasis": float(sre),
        "glrlm_long_run_emphasis": float(lre),
        "glrlm_gray_level_nonuniformity": float(gln),
        "glrlm_run_length_nonuniformity": float(rln)
    }

class RadiomicFeatureExtractor:
    """
    Extracts radiomic features from imaging data in a registered folder.
    Expects the following file names inside the folder:
      - T1: T1_corrected_canonical.nii_toMag.nii.gz
      - T2: T2_FLAIR_corrected_canonical.nii_toMag.nii.gz
      - QSM: QSM_canonical.nii.gz
      - Mask: lesions_MSpace_Mask.nii.gz
    This extractor treats each lesion independently. For each unique lesion label
    (excluding background), it computes:
      - Basic intensity features (mean, std) for T1, T2, QSM over the entire 3D lesion.
      - Geometric features (lesion volume in voxels and physical volume).
      - Texture features (GLCM and GLRLM) computed on each 2D slice containing the lesion,
        then averaged across slices.
    """
    def __init__(self, registered_folder):
        self.registered_folder = registered_folder
        self.t1_path = os.path.join(registered_folder, "T1_corrected_canonical.nii_toMag.nii.gz")
        self.t2_path = os.path.join(registered_folder, "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz")
        self.qsm_path = os.path.join(registered_folder, "QSM_canonical.nii.gz")
        self.mask_path = os.path.join(registered_folder, "lesions_MSpace_Mask.nii.gz")
        
        self.t1_data = None
        self.t2_data = None
        self.qsm_data = None
        self.mask_data = None
        self.voxel_volume = None

    def load_images(self):
        """Load all images and compute the voxel volume using the mask image affine."""
        self.t1_data = nib.load(self.t1_path).get_fdata()
        self.t2_data = nib.load(self.t2_path).get_fdata()
        self.qsm_data = nib.load(self.qsm_path).get_fdata()
        mask_img = nib.load(self.mask_path)
        self.mask_data = mask_img.get_fdata()
        self.voxel_volume = np.abs(np.linalg.det(mask_img.affine[:3, :3]))


    def extract_features(self):
        """
        Extract features for each lesion in the 3D mask. For texture features, all slices where the lesion
        is present are processed and their features are averaged for each modality.
        
        Returns
        -------
        list of dict
            Each dict contains features for one lesion, with texture features for each modality in separate columns.
        """
        if any(img is None for img in [self.t1_data, self.t2_data, self.qsm_data, self.mask_data]):
            raise ValueError("Images not loaded. Call load_images() first.")

        features_list = []
        unique_labels = np.unique(self.mask_data)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background

        for label_id in unique_labels:
            lesion_mask = (self.mask_data == label_id)
            num_voxels = int(np.count_nonzero(lesion_mask))
            volume_phys = num_voxels * self.voxel_volume

            # Basic intensity features over the entire 3D lesion
            t1_vals = self.t1_data[lesion_mask]
            t2_vals = self.t2_data[lesion_mask]
            qsm_vals = self.qsm_data[lesion_mask]
            basic_feats = {
                "label_id": int(label_id),
                "num_voxels": num_voxels,
                "volume_physical": volume_phys,
                "T1_mean": float(np.mean(t1_vals)),
                "T1_std": float(np.std(t1_vals)),
                "T2_mean": float(np.mean(t2_vals)),
                "T2_std": float(np.std(t2_vals)),
                "QSM_mean": float(np.mean(qsm_vals)),
                "QSM_std": float(np.std(qsm_vals))
            }

            # Additional first order features for each modality
            t1_first_order = compute_first_order_features(t1_vals, modality="T1")
            t2_first_order = compute_first_order_features(t2_vals, modality="T2")
            qsm_first_order = compute_first_order_features(qsm_vals, modality="QSM")
            
            # --- Texture features computed from each 2D slice containing part of the lesion ---
            # Create separate lists for each modality.
            t1_glcm_list, t1_glrlm_list = [], []
            t2_glcm_list, t2_glrlm_list = [], []
            qsm_glcm_list, qsm_glrlm_list = [], []
            
            # Determine the z-slices where the lesion is present.
            z_indices = np.unique(np.argwhere(lesion_mask)[:, 2])
            for z in z_indices:
                slice_mask = lesion_mask[:, :, z]
                if np.count_nonzero(slice_mask) == 0: # Edge case: no lesion in this slice
                    continue
                # Get the bounding box for the lesion in this slice.
                # We isolate the lesion within a specific slice of the mask.
                #Defines the smallest rectangular region that contains all the lesion pixels in that slice
                coords = np.argwhere(slice_mask)
                if coords.size == 0:
                    continue
                min_xy = coords.min(axis=0)
                max_xy = coords.max(axis=0)
                
                # For each modality, extract the 2D patch from the corresponding slice.
                # To compute textures on the 2D slice.
                # We capture slice specific texture information
                # 
                patch_t1 = self.t1_data[min_xy[0]:max_xy[0] + 1, min_xy[1]:max_xy[1] + 1, z]
                patch_t2 = self.t2_data[min_xy[0]:max_xy[0] + 1, min_xy[1]:max_xy[1] + 1, z]
                patch_qsm = self.qsm_data[min_xy[0]:max_xy[0] + 1, min_xy[1]:max_xy[1] + 1, z]
                
                # Quantize each patch.
                patch_t1_quant = quantize_image(patch_t1, levels=256)
                patch_t2_quant = quantize_image(patch_t2, levels=256)
                patch_qsm_quant = quantize_image(patch_qsm, levels=256)

                plt.hist(patch_qsm_quant.ravel(), bins=256, range=(0, 255))
                plt.title(f"Histogram of quantized image")
                plt.xlabel("Intensity")
                plt.ylabel("Frequency")
                plt.savefig(f"histogram_{label_id}_slice_{z}.png")

                
                # Compute texture features for each modality.
                t1_glcm_feats = compute_glcm_features_2d(patch_t1_quant)
                t1_glrlm_feats = compute_glrlm_features_2d(patch_t1_quant)
                t2_glcm_feats = compute_glcm_features_2d(patch_t2_quant)
                t2_glrlm_feats = compute_glrlm_features_2d(patch_t2_quant)
                qsm_glcm_feats = compute_glcm_features_2d(patch_qsm_quant)
                qsm_glrlm_feats = compute_glrlm_features_2d(patch_qsm_quant)
                
                # Append the computed features to the corresponding modality lists.
                t1_glcm_list.append(t1_glcm_feats)
                t1_glrlm_list.append(t1_glrlm_feats)
                t2_glcm_list.append(t2_glcm_feats)
                t2_glrlm_list.append(t2_glrlm_feats)
                qsm_glcm_list.append(qsm_glcm_feats)
                qsm_glrlm_list.append(qsm_glrlm_feats)
            
            # Average the texture features across slices for each modality.

            glcm_feature_keys = [
                "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
                "glcm_energy", "glcm_correlation", "glcm_asm",
                "glcm_entropy", "glcm_max_probability",
                "glcm_cluster_shade", "glcm_cluster_prominence"
            ]

            glrlm_feature_keys = [
                "glrlm_mean_run", "glrlm_short_run_emphasis", "glrlm_long_run_emphasis",
                "glrlm_gray_level_nonuniformity", "glrlm_run_length_nonuniformity"
            ]   
            # GLCM features:
            if t1_glcm_list:
                avg_t1_glcm = {f"T1_{k}": np.mean([d[k] for d in t1_glcm_list]) for k in t1_glcm_list[0]}
            else:
                avg_t1_glcm = {f"T1_{k}": np.nan for k in glcm_feature_keys}
            if t2_glcm_list:
                avg_t2_glcm = {f"T2_{k}": np.mean([d[k] for d in t2_glcm_list]) for k in t2_glcm_list[0]}
            else:
                avg_t2_glcm = {f"T2_{k}": np.nan for k in glcm_feature_keys}
            if qsm_glcm_list:
                avg_qsm_glcm = {f"QSM_{k}": np.mean([d[k] for d in qsm_glcm_list]) for k in qsm_glcm_list[0]}
            else:
                avg_qsm_glcm = {f"QSM_{k}": np.nan for k in glcm_feature_keys}
                
            # GLRLM features:
            if t1_glrlm_list:
                avg_t1_glrlm = {f"T1_{k}": np.mean([d[k] for d in t1_glrlm_list]) for k in t1_glrlm_list[0]}
            else:
                avg_t1_glrlm = {f"T1_{k}": np.nan for k in glrlm_feature_keys}
            if t2_glrlm_list:
                avg_t2_glrlm = {f"T2_{k}": np.mean([d[k] for d in t2_glrlm_list]) for k in t2_glrlm_list[0]}
            else:
                avg_t2_glrlm = {f"T2_{k}": np.nan for k in glrlm_feature_keys}
            if qsm_glrlm_list:
                avg_qsm_glrlm = {f"QSM_{k}": np.mean([d[k] for d in qsm_glrlm_list]) for k in qsm_glrlm_list[0]}
            else:
                avg_qsm_glrlm = {f"QSM_{k}": np.nan for k in glrlm_feature_keys}
            
            # Combine all averaged texture features for all modalities.
            texture_feats = {**avg_t1_glcm, **avg_t1_glrlm,
                            **avg_t2_glcm, **avg_t2_glrlm,
                            **avg_qsm_glcm, **avg_qsm_glrlm}
            
            # Combine all features: basic 3D features, first order features, and texture features.
            all_feats = {**basic_feats, **t1_first_order, **t2_first_order, **qsm_first_order, **texture_feats}
            features_list.append(all_feats)
            
        return features_list
