#!/usr/bin/env python3
import argparse
import pandas as pd

def count_between_thresholds(df, threshold_pairs):
    """
    For each (min_thr, max_thr) in threshold_pairs, count lesions with
    num_voxels >= min_thr and num_voxels <= max_thr.
    Returns a DataFrame with columns ['min_voxels', 'max_voxels', 'lesion_count', 'percent'].
    """
    records = []
    total = len(df)
    for min_thr, max_thr in threshold_pairs:
        subset = df[(df["num_voxels"] >= min_thr) & (df["num_voxels"] <= max_thr)]
        cnt = len(subset)
        percent = cnt / total * 100 if total > 0 else 0
        records.append({
            "min_voxels": min_thr,
            "max_voxels": max_thr,
            "lesion_count": cnt,
            "percent": percent
        })
    return pd.DataFrame.from_records(records)

def group_by_quantiles(df, n_groups):
    """
    Divide lesions into n_groups based on num_voxels quantiles.
    Returns the original df with an added 'volume_group' column and
    a summary DataFrame with group boundaries and counts.
    """
    df = df.copy()
    # Assign group numbers 1..n based on quantiles
    df['volume_group'] = pd.qcut(df['num_voxels'], q=n_groups, labels=False, duplicates='drop') + 1
    
    # Determine quantile intervals
    intervals = pd.qcut(df['num_voxels'], q=n_groups, duplicates='drop').unique().categories
    summary = []
    total = len(df)
    for idx, interval in enumerate(intervals, start=1):
        group_df = df[(df['num_voxels'] >= interval.left) & (df['num_voxels'] <= interval.right)]
        summary.append({
            "group": idx,
            "lower_bound": interval.left,
            "upper_bound": interval.right,
            "lesion_count": len(group_df),
            "percent": len(group_df) / total * 100 if total > 0 else 0
        })
    summary_df = pd.DataFrame(summary)
    return df, summary_df

def main():
    parser = argparse.ArgumentParser(
        description="Count and group lesions by volume thresholds and quantiles."
    )
    parser.add_argument("csv_file",
                        help="CSV file with a 'num_voxels' column.")
    parser.add_argument("--min_thr", "-l", nargs="+", type=int, default=None,
                        help="List of lower thresholds (e.g. -l 0 50 100).")
    parser.add_argument("--max_thr", "-u", nargs="+", type=int, default=None,
                        help="List of upper thresholds, matching --min_thr.")
    parser.add_argument("--n_groups", "-g", type=int, default=None,
                        help="Number of quantile groups to create (e.g. -g 4).")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    if "num_voxels" not in df.columns:
        parser.error("CSV must contain a 'num_voxels' column")

    # Threshold-based counts
    if args.min_thr and args.max_thr:
        if len(args.min_thr) != len(args.max_thr):
            parser.error("Provide equal numbers of --min_thr and --max_thr")
        threshold_pairs = list(zip(args.min_thr, args.max_thr))
        thr_counts = count_between_thresholds(df, threshold_pairs)
        print("Counts between thresholds:")
        print(thr_counts.to_string(index=False))
        print()

    # Quantile grouping
    if args.n_groups:
        grouped_df, summary_df = group_by_quantiles(df, args.n_groups)
        print(f"Quantile grouping into {args.n_groups} bins:")
        print(summary_df.to_string(index=False))
        print()

if __name__ == "__main__":
    main()
