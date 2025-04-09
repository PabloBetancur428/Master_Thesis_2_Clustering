"""
feature_extraction.py

Module for extracting radiomic features from a 3D multi-labeled lesion mask using PyRadiomics.
For each lesion (each unique nonzero label in the mask), features are computed using both:
  - 3D extraction (entire lesion as a volume).
  - 2D extraction (slice-based extraction).
Additionally, by adding a wavelet filter into the parameters, features are also computed on
wavelet-decomposed images. The results are stored with modality, extraction-type and filter
prefixes.
"""

import os
import numpy as np
import nibabel as nib
from radiomics import featureextractor

class RadiomicFeatureExtractor:
    def __init__(self, registered_folder, params=None):
        """
        Initialize the feature extractor.
        
        Parameters
        ----------
        registered_folder : str
            Path to the folder containing the registered images.
            Expected files:
              - T1_corrected_canonical.nii_toMag.nii.gz
              - T2_FLAIR_corrected_canonical.nii_toMag.nii.gz
              - QSM_canonical.nii.gz
              - lesions_MSpace_Mask.nii.gz
        params : dict or str, optional
            Parameter settings for PyRadiomics. Can be provided as a dict or a path to a YAML file.
            If None, default settings are used. Two sets of default settings are created:
                one for 3D extraction (force2D=False) and one for 2D (force2D=True).
            Both include a wavelet transform filter under "imageType".
        """
        self.registered_folder = registered_folder
        self.t1_path = os.path.join(registered_folder, "T1_corrected_canonical.nii_toMag.nii.gz")
        self.t2_path = os.path.join(registered_folder, "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz")
        self.qsm_path = os.path.join(registered_folder, "QSM_canonical.nii.gz")
        self.mask_path = os.path.join(registered_folder, "lesions_MSpace_Mask.nii.gz")
        
        # If no custom parameters are provided, use default settings that include a wavelet filter.
        if params is None:
            default_params_3d = {
                "force2D": False,
                "imageType": {
                    "Original": {},
                    "Wavelet": {}  # Enable wavelet decomposition for additional filtered features.
                }
            }
            default_params_2d = {
                "force2D": True,
                "imageType": {
                    "Original": {},
                    "Wavelet": {}
                }
            }
            self.extractor_3d = featureextractor.RadiomicsFeatureExtractor(**default_params_3d)
            self.extractor_2d = featureextractor.RadiomicsFeatureExtractor(**default_params_2d)
        else:
            # When custom parameters are provided, override the force2D parameter as needed:
            if isinstance(params, dict):
                params_3d = params.copy()
                params_3d["force2D"] = False
                # Ensure wavelet is enabled if not already included.
                params_3d.setdefault("imageType", {}) 
                params_3d["imageType"].setdefault("Wavelet", {})
                
                params_2d = params.copy()
                params_2d["force2D"] = True
                params_2d.setdefault("imageType", {})
                params_2d["imageType"].setdefault("Wavelet", {})
            else:
                # If a YAML file is provided, we'll assume it contains appropriate settings.
                params_3d = params
                params_2d = params
                
            self.extractor_3d = featureextractor.RadiomicsFeatureExtractor(params_3d)
            self.extractor_2d = featureextractor.RadiomicsFeatureExtractor(params_2d)
        
        # Placeholders for mask data and voxel volume.
        self.mask_data = None
        self.voxel_volume = None
        
    def load_mask(self):
        """Load the mask image and compute the voxel volume using the affine from the NIfTI header."""
        mask_img = nib.load(self.mask_path)
        self.mask_data = mask_img.get_fdata()
        self.voxel_volume = np.abs(np.linalg.det(mask_img.affine[:3, :3]))
        
    def extract_features(self):
        """
        Extract features for each lesion in the 3D mask using both 3D and 2D PyRadiomics extractors.
        For each unique lesion label (excluding background), features are computed for each modality.
        Additionally, with the wavelet filter in effect, features extracted from wavelet-decomposed images
        will also be included.
        
        Returns
        -------
        list of dict
            Each dictionary contains features for one lesion:
             - 3D features are prefixed with <modality>_3d_.
             - 2D features are prefixed with <modality>_2d_.
             - Wavelet filtered features will have additional keys (e.g., waveletOriginal, waveletWavelet, etc.)
            Basic geometric info (e.g., num_voxels and volume_physical) is also included.
        """
        # Ensure that the mask is loaded.
        if self.mask_data is None:
            self.load_mask()
            
        features_list = []
        unique_labels = np.unique(self.mask_data)
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background
        
        for label_id in unique_labels:
            lesion_features = {"label_id": int(label_id)}
            
            # Compute basic geometric features from the lesion mask.
            lesion_mask = (self.mask_data == label_id)
            num_voxels = int(np.count_nonzero(lesion_mask))
            volume_phys = num_voxels * self.voxel_volume
            lesion_features["num_voxels"] = num_voxels
            lesion_features["volume_physical"] = volume_phys
            
            # Define the modalities and their respective image paths.
            modalities = {
                "T1": self.t1_path,
                "T2": self.t2_path,
                "QSM": self.qsm_path
            }
            
            # For each modality, extract features using both the 3D and 2D extractors.
            for modality, image_path in modalities.items():
                # 3D extraction.
                try:
                    result_3d = self.extractor_3d.execute(image_path, self.mask_path, label=int(label_id))
                    for key, value in result_3d.items():
                        if key.startswith("diagnostics_"):
                            continue
                        new_key = f"{modality}_3d_{key}"
                        lesion_features[new_key] = value
                except Exception as e:
                    lesion_features[f"{modality}_3d_error"] = str(e)
                    
                # 2D extraction.
                try:
                    result_2d = self.extractor_2d.execute(image_path, self.mask_path, label=int(label_id))
                    for key, value in result_2d.items():
                        if key.startswith("diagnostics_"):
                            continue
                        new_key = f"{modality}_2d_{key}"
                        lesion_features[new_key] = value
                except Exception as e:
                    lesion_features[f"{modality}_2d_error"] = str(e)
                    
            features_list.append(lesion_features)
            
        return features_list

# Example usage:
if __name__ == "__main__":
    # Replace '/path/to/registered_folder' with the correct path where your images are located.
    registered_folder = "/path/to/registered_folder"
    extractor = RadiomicFeatureExtractor(registered_folder)
    features = extractor.extract_features()
    print(features)
