"""
feature_extraction.py

Module to extract lesion volumes from a multi-labeled mask.
For a given nifti mask, it computes:
    -The voxel volume (using the affine matrix)
    -For each lesion -> the number of voxels times the voxel volume

"""

import nibabel as nib
import numpy as np

class LesionVolumeExtractor:
    def __init__(self, mask_path):
        """
        Initialize with path to the multi-labeled lesion mask.

        Parameters
        ---------
        mask_path: str
            Path to the lesion mask NifTI file
        """
        self.mask_path = mask_path
        self.mask_img = None
        self.mask_data = None
        self.voxel_volume = None

    def load_mask(self):
        """Load the mask image and compute the voxel volume."""
        self.mask_img = nib.load(self.mask_path)
        self.mask_data = self.mask_img.get_fdata()
        # Compute voxel volume using the absolute value of the determinant of the 3x3 part of the affine.
        self.voxel_volume = np.abs(np.linalg.det(self.mask_img.affine[:3, :3]))

    def extract_lesion_volumes(self):
        """
        Compute the volume for each lesion.
        
        Returns
        -------
        dict
            Dictionary mapping lesion label (int) to its physical volume (float).
        """
        if self.mask_data is None or self.voxel_volume is None:
            raise ValueError("Mask data not loaded. Call load_mask() first.")

        lesion_volumes = {}
        unique_labels = np.unique(self.mask_data)
        # Exclude the background (assumed to be label 0)
        for label_id in unique_labels:
            if label_id == 0:
                continue
            # Count voxels belonging to this lesion label
            count = np.count_nonzero(self.mask_data == label_id)
            # Multiply by voxel volume to get physical volume
            lesion_volumes[int(label_id)] = count * self.voxel_volume
        return lesion_volumes