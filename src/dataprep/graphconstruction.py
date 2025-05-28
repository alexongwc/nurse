# src/dataprep/graphconstruction.py

import pandas as pd
import torch
from torch_geometric.data import Data

def build_gat_graph(edge_csv_path):
    """
    Load edge dataframe and build a PyTorch Geometric Data object for GAT.
    
    Args:
        edge_csv_path (str): Path to the combined edge list CSV.
        
    Returns:
        data (torch_geometric.data.Data): Graph object for GAT.
        nurse2idx (dict): Nurse ID to node index mapping.
        shift2idx (dict): Shift ID to node index mapping.
    """
    # Load edge list
    edge_df = pd.read_csv(edge_csv_path)

    # Create unique shift IDs
    edge_df['shift_id'] = edge_df['date'].astype(str) + "_" + edge_df['shift'] + "_" + edge_df['ward']

    # Build nurse and shift mappings
    nurse2idx = {n: i for i, n in enumerate(edge_df['nurse_id'].unique())}
    shift2idx = {s: i + len(nurse2idx) for i, s in enumerate(edge_df['shift_id'].unique())}

    # Add node indices to dataframe
    edge_df['nurse_idx'] = edge_df['nurse_id'].map(nurse2idx)
    edge_df['shift_idx'] = edge_df['shift_id'].map(shift2idx)

    # Build edge index
    edge_index = torch.tensor([
        edge_df['nurse_idx'].values,
        edge_df['shift_idx'].values
    ], dtype=torch.long)

    # Build node features: type encoding (nurse=0, shift=1)
    num_nurses = len(nurse2idx)
    num_shifts = len(shift2idx)
    num_nodes = num_nurses + num_shifts
    x = torch.zeros((num_nodes, 1))
    x[num_nurses:, 0] = 1

    # Labels (for edge-level supervision)
    y = torch.tensor(edge_df['label'].values, dtype=torch.float)

    # Optional: edge features can be added as needed
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=None,  # Add edge features here if you wish
        y=y
    )
    return data, nurse2idx, shift2idx

# Example usage
if __name__ == "__main__":
    data, nurse2idx, shift2idx = build_gat_graph("../data/edges_for_gat.csv")
    print(data)
    print("Number of nodes:", data.num_nodes)
    print("Number of edges:", data.num_edges)
    print("Node features shape:", data.x.shape)
