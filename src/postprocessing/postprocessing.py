#src/postprocessing/postprocessing.py

import pandas as pd
import os

def prepare_edges_for_assignment(
    edges_path="data/edges_with_scores.csv",
    nurse_map_path="data/nurse_idx_map.csv",
    shift_map_path="data/shift_idx_map.csv",
    output_path="data/edges_with_scores_readable.csv"
):
    """
    Combines model output (edges_with_scores.csv) with nurse and shift mappings
    to produce a human-readable edge list for the assignment solver/UI.
    """
    # 1. Load files
    df_edges = pd.read_csv(edges_path)
    df_nurse = pd.read_csv(nurse_map_path)
    df_shift = pd.read_csv(shift_map_path)

    # 2. Merge nurse and shift mapping
    df = df_edges.merge(df_nurse, left_on='nurse_node', right_on='nurse_idx', how='left')
    df = df.merge(df_shift, left_on='shift_node', right_on='shift_idx', how='left')

    # 3. Ensure duration_hours is included!
    if 'duration_hours' not in df.columns:
        print("WARNING: 'duration_hours' missing after merge. Trying to patch from original data...")
        df_orig = pd.read_csv("data/combined.csv")
        df_orig['shift_id'] = (
            df_orig['date'].astype(str) + "_" +
            df_orig['shift'].astype(str) + "_" +
            df_orig['ward'].astype(str) + "_" +
            df_orig['start_time'].astype(str) + "_" +
            df_orig['end_time'].astype(str)
        )
        df = df.merge(
            df_orig[['shift_id', 'duration_hours']].drop_duplicates(),
            on='shift_id',
            how='left'
        )

    print("Final columns in merged DataFrame:", df.columns.tolist())

    # 4. Output columns for assignment solver and UI
    columns_out = [
        'nurse_node', 'nurse_id',
        'shift_node', 'date', 'shift', 'ward', 'start_time', 'end_time',
        'duration_hours', 'gat_score'
    ]
    columns_out = [col for col in columns_out if col in df.columns]
    df = df[columns_out]

    # 5. Save
    df.to_csv(output_path, index=False)
    print(f"[postprocessing] Human-readable edge list saved to {output_path}")
    return output_path

if __name__ == "__main__":
    prepare_edges_for_assignment()