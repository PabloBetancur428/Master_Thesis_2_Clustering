import os
import pandas as pd

def check_patients_in_folder(
    excel_file, 
    folder_path, 
    id_col="PatientID", 
    value_col="SomeColumn"
):
    """
    Reads an Excel file, checks if each patient ID has a corresponding folder
    in 'folder_path', and prints the ID alongside a specified column value.

    Parameters
    ----------
    excel_file : str
        Path to the .xlsx file.
    folder_path : str
        Directory in which we check for folders named after patient IDs.
    id_col : str
        The name of the column in the Excel file containing patient IDs.
    value_col : str
        The name of the column whose value we want to print next to the ID.

    Returns
    -------
    None
        Prints the ID and the column value for each matching folder.
    """
    # 1) Read the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)
    

    pts = 0
    controls = 0
    what = 0
    processed_ids = set() #to keep track of processed pt IDs
    excel_patient_ids = set(df[id_col].astype(str))


    for idx, row in df.iterrows():
        patient_id = str(row[id_col])

        # SKip if this patient ID has already been processed
        if patient_id in processed_ids:
            continue

        patient_folder = os.path.join(folder_path, patient_id)

        
        if os.path.isdir(patient_folder):
            what += 1
            print(f"Found patient folder: {patient_id}")
            print(f"==>{value_col}: {row[value_col]}")

            if row[value_col] == 1:
                pts += 1
            elif row[value_col] == 0:
                controls += 1
            else:
                print(f"ERROR: Invalid value in column '{value_col}' for patient ID {patient_id}")
                break

            processed_ids.add(patient_id)
    
    #Identify folder that are not in the excel
    all_folders= { f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))}
    missing_folders = all_folders - excel_patient_ids
    
    
    print(f"Total patients (val = 1): {pts}")
    print(f"Total controls (val = 0): {controls}")
    print(f"Total processed IDs: {len(processed_ids)}")
    print(f"Total IDs in the Excel file: {len(df)}")
    print(f"Total people: {pts + controls}")
    print(f"Folders present in the directory but not listed in the excel file: {missing_folders}")

# Example usage
if __name__ == "__main__":
    excel_file_path = "/home/jbetancur/Desktop/codes/clustering/data/Metadata/25-11-2025_xnatt_brain.xlsx"
    directory_path_baseline  = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_baseline"
    directory_path_follow_up = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/clean_follow_up"

    print("Patients vs Controls baseline")
    check_patients_in_folder(
        excel_file=excel_file_path,
        folder_path=directory_path_baseline,
        id_col="subject_id",
        value_col="patient"
    )

    print("Patients vs control follow-up")
    check_patients_in_folder(
    excel_file=excel_file_path,
    folder_path=directory_path_follow_up,
    id_col="subject_id",
    value_col="patient"
    )

