import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_and_explore_dataframe(df, output_dir='./output', filename_prefix=''):
    """
    Saves the input DataFrame to CSV files and produces a series of plots for data exploration.
    Expects the DataFrame to have at least:
      - A 'PatientID' column (for grouping lesions per patient)
      - A 'LesionVolume' column (for the volume of each lesion)
      
    The function:
      1. Saves the full DataFrame to CSV.
      2. Computes and saves basic summary statistics for lesion volumes.
      3. Generates a comprehensive plot with:
         - Histogram, Box Plot, and Violin Plot of lesion volumes.
         - A distribution plot of the number of lesions per patient.
      4. Computes advanced stats (skewness and kurtosis) and detects outliers.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file names with the provided prefix
    df_csv = os.path.join(output_dir, f'{filename_prefix}lesion_volumes.csv')
    summary_csv = os.path.join(output_dir, f'{filename_prefix}volume_summary_statistics_all.csv')
    plot_path = os.path.join(output_dir, f'{filename_prefix}lesion_volume_distribution_comprehensive_all.png')
    outliers_csv = os.path.join(output_dir, f'{filename_prefix}lesion_outliers_all.csv')

    # 1. Save DataFrame to CSV
    df.to_csv(df_csv, index=False)
    print(f"DataFrame saved to {df_csv}")

    # 2. Basic Statistical Summary for 'LesionVolume'
    if 'LesionVolume' not in df.columns:
        raise ValueError("DataFrame must contain a 'LesionVolume' column")
    summary_stats = df['LesionVolume'].describe()
    summary_stats.to_csv(summary_csv)
    print("Summary Statistics:\n", summary_stats)

    # 3. Comprehensive Visualization
    plt.figure(figsize=(15, 10))

    # Histogram of Lesion Volumes
    plt.subplot(2, 2, 1)
    sns.histplot(df['LesionVolume'], kde=True, bins=63)
    plt.title('Histogram of Lesion Volumes')
    plt.xlabel('Lesion Volume')

    # Box Plot of Lesion Volumes
    plt.subplot(2, 2, 2)
    sns.boxplot(x=df['LesionVolume'])
    plt.title('Box Plot of Lesion Volumes')

    # Violin Plot of Lesion Volumes
    plt.subplot(2, 2, 3)
    sns.violinplot(x=df['LesionVolume'])
    plt.title('Violin Plot of Lesion Volumes')

    # Distribution of Number of Lesions per Patient
    plt.subplot(2, 2, 4)
    lesion_counts = df.groupby("PatientID").size()
    sns.histplot(lesion_counts, kde=True, bins=60)
    plt.title('Number of Lesions per Patient')
    plt.xlabel('Number of Lesions')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Comprehensive plot saved to {plot_path}")
    plt.show()

    # 4. Advanced Statistical Analysis: Skewness and Kurtosis
    skewness = df['LesionVolume'].skew()
    kurtosis = df['LesionVolume'].kurtosis()
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}")

    # 5. Outlier Detection using IQR on LesionVolume
    Q1 = df['LesionVolume'].quantile(0.25)
    Q3 = df['LesionVolume'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (df['LesionVolume'] < (Q1 - 1.5 * IQR)) | (df['LesionVolume'] > (Q3 + 1.5 * IQR))
    outliers = df[outlier_condition]
    outliers.to_csv(outliers_csv, index=False)
    print("Number of outlier lesions:", len(outliers))
    print("Outliers:\n", outliers)

    return df
