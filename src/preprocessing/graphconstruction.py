# src/dataprep/graphconstruction.py

import torch
from torch_geometric.data import Data

def build_graph(edge_df, save_mapping_dir="src"):
    edge_df = edge_df.copy()

    # Build unique shift_id using all relevant columns
    if 'shift_id' not in edge_df.columns:
        edge_df['shift_id'] = (
            edge_df['date'].astype(str) + "_" +
            edge_df['shift'].astype(str) + "_" +
            edge_df['ward'].astype(str) + "_" +
            edge_df['start_time'].astype(str) + "_" +
            edge_df['end_time'].astype(str)
        )

    # Nurse and shift index mapping
    nurse2idx = {n: i for i, n in enumerate(edge_df['nurse_id'].unique())}
    shift2idx = {s: i + len(nurse2idx) for i, s in enumerate(edge_df['shift_id'].unique())}

    # Assign indices
    edge_df['nurse_idx'] = edge_df['nurse_id'].map(nurse2idx)
    edge_df['shift_idx'] = edge_df['shift_id'].map(shift2idx)

    # Edge index for PyG
    edge_index = torch.tensor([
        edge_df['nurse_idx'].values,
        edge_df['shift_idx'].values
    ], dtype=torch.long)

    num_nurses = len(nurse2idx)
    num_shifts = len(shift2idx)
    num_nodes = num_nurses + num_shifts

    # Node type encoding: 0=nurse, 1=shift
    x = torch.zeros((num_nodes, 1))
    x[num_nurses:, 0] = 1

    y = torch.tensor(edge_df['label'].values, dtype=torch.float)

    # ----- Save index mappings for postprocessing -----
    import os
    import pandas as pd

    # Save nurse mapping
    nurse_idx_map = (
        edge_df[['nurse_idx', 'nurse_id']]
        .drop_duplicates()
        .sort_values('nurse_idx')
    )
    nurse_map_path = os.path.join(save_mapping_dir, "nurse_idx_map.csv")
    nurse_idx_map.to_csv(nurse_map_path, index=False)

    # Save shift mapping (now includes duration_hours)
    shift_idx_map = (
        edge_df[['shift_idx', 'shift_id', 'date', 'shift', 'ward', 'start_time', 'end_time', 'duration_hours']]
        .drop_duplicates()
        .sort_values('shift_idx')
    )
    shift_map_path = os.path.join(save_mapping_dir, "shift_idx_map.csv")
    shift_idx_map.to_csv(shift_map_path, index=False)

    print(f"Saved nurse mapping to {nurse_map_path}")
    print(f"Saved shift mapping to {shift_map_path}")

    # ----- Return PyG Data object -----
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=None,
        y=y
    )
    return data
