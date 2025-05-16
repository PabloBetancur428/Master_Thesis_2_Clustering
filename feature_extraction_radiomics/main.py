import os
import logging
import pandas as pd
from data_loader_folders import load_all_patient_files
from pyradiomics_feature_extraction import RadiomicFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main():
    baseline_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_baseline"
    followup_dir = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_follow_up"
    excel_file   = "/home/jbetancur/Desktop/codes/clustering/patients_with_qsm.xlsx"

    cfg_dir = os.path.join(os.path.dirname(__file__), "config")
    params3d = os.path.join(cfg_dir, "radiomics_params_3d.yaml")
    params2d = os.path.join(cfg_dir, "radiomics_params_2d.yaml")

    logging.info("Loading patient records…")
    patient_records = load_all_patient_files(baseline_dir, followup_dir, excel_file)
    if not patient_records:
        logging.warning("No patient records found; exiting.")
        return
    logging.info(f"Found {len(patient_records)} patients.")

    all_feats = []
    for rec in patient_records:
        pid, ftype, year, reg = rec["PatientID"], rec["FolderType"], rec["Year"], os.path.dirname(rec["T1"])
        logging.info(f"Processing {pid} ({ftype}, {year})")
        try:
            extr = RadiomicFeatureExtractor(reg, params3d=params3d, params2d=params2d)
            feats = extr.extract_features()
        except Exception as e:
            logging.error(f"Failed on {pid}: {e}")
            continue
        logging.info(f"Extracted {len(feats)} lesion feature‐sets for {pid}")
        for f in feats:
            f.update({"PatientID": pid, "FolderType": ftype, "Year": year})
            all_feats.append(f)

    df = pd.DataFrame(all_feats)
    out_dir = os.path.join("feature_extraction", "output")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "Radiomic_QSM_Wavelet_LBP_2d_3d.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved to {csv_path}")

if __name__ == "__main__":
    main()
