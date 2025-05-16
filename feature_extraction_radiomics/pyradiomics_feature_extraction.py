import os
import numpy as np
import nibabel as nib
from radiomics.featureextractor import RadiomicsFeatureExtractor as PyradExtractor

class RadiomicFeatureExtractor:
    def __init__(self,
                 registered_folder: str,
                 params3d: str,
                 params2d: str,
                 min_voxels: int = 12,
                 max_voxels: int = 10000):

        self.registered_folder = registered_folder
        self.min_voxels = min_voxels
        self.max_voxels = max_voxels

        # image paths
        self.t1_path   = os.path.join(registered_folder, "T1_corrected_canonical.nii_toMag.nii.gz")
        self.t2_path   = os.path.join(registered_folder, "T2_FLAIR_corrected_canonical.nii_toMag.nii.gz")
        self.qsm_path  = os.path.join(registered_folder, "QSM_canonical.nii.gz")
        self.mask_path = os.path.join(registered_folder, "lesions_MSpace_Mask.nii.gz")

        # **Load** YAML configs by passing them as the *first* positional arg
        self.ext3d = PyradExtractor(params3d)
        self.ext2d = PyradExtractor(params2d)

        self.mask_data = None
        self.voxel_volume = None

    def load_mask(self):
        mask_img = nib.load(self.mask_path)
        self.mask_data = mask_img.get_fdata()
        self.voxel_volume = abs(np.linalg.det(mask_img.affine[:3, :3]))

    def extract_features(self):
        if self.mask_data is None:
            self.load_mask()

        features_list = []
        labels = np.unique(self.mask_data).astype(int)
        labels = labels[labels != 0]

        for lbl in labels:
            mask = (self.mask_data == lbl)
            nvx = int(np.count_nonzero(mask))
            if nvx < self.min_voxels or (self.max_voxels and nvx > self.max_voxels):
                continue

            phys_vol = nvx * self.voxel_volume
            feat = {
                "label_id": lbl,
                "num_voxels": nvx,
                "volume_physical": phys_vol
            }

            for mod_name, img_path in (
                ("T1", self.t1_path),
                ("T2", self.t2_path),
                ("QSM", self.qsm_path),
            ):
                # 3D
                try:
                    print("TRyiiiing")
                    out3d = self.ext3d.execute(img_path, self.mask_path, label=lbl)
                    for k, v in out3d.items():
                        if not k.startswith("diagnostics_"):
                            feat[f"{mod_name}_3d_{k}"] = v
                except Exception as e:
                    print("Puta madre")
                    feat[f"{mod_name}_3d_error"] = str(e)

                # 2D
                try:
                    print("TRyiiiing")
                    out2d = self.ext2d.execute(img_path, self.mask_path, label=lbl)
                    for k, v in out2d.items():
                        if not k.startswith("diagnostics_"):
                            feat[f"{mod_name}_2d_{k}"] = v
                except Exception as e:
                    print("Puta madre")
                    feat[f"{mod_name}_2d_error"] = str(e)

            features_list.append(feat)

        return features_list
