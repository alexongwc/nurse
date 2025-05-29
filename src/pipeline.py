# pipeline.py
# pipeline.py

import pandas as pd
import yaml
import os
import itertools

from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.graphconstruction import build_graph
from src.model.model import train_gat, predict_gat

from src.postprocessing.postprocessing import prepare_edges_for_assignment
from src.postprocessing.assignmentsolver import solve_assignment

# --- Load config ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = "data"

# --- Step 1: Data prep (all history for training) ---
df = pd.read_csv(cfg['paths']['combined_csv'])
df['date'] = pd.to_datetime(df['date'])
df = feature_engineering(df)
train_assigned = df[df['label'] == 1].copy()

# --- Step 2: Generate candidates for live period (preference.csv) ---
PREFERENCE_CSV = os.path.join(DATA_DIR, "preference.csv")
pref_df = pd.read_csv(PREFERENCE_CSV)
pref_df['date'] = pd.to_datetime(pref_df['date'])

# Standardize column names for easier downstream processing
pref_df = pref_df.rename(columns={
    'preferred_shift': 'shift',
    'preferred_ward': 'ward'
})

shift_cols = ['date', 'shift', 'ward', 'start_time', 'end_time', 'duration_hours']
shifts_for_window = pref_df[shift_cols].drop_duplicates().reset_index(drop=True)

nurse_list = df['nurse_id'].unique()

candidates = list(itertools.product(nurse_list, shifts_for_window.index))
candidate_df = pd.DataFrame(candidates, columns=['nurse_id', 'shift_idx'])
candidate_df = candidate_df.merge(shifts_for_window, left_on='shift_idx', right_index=True, how='left')

# Optionally flag if nurse actually requested this shift
def is_preferred(row):
    return ((pref_df['nurse_id'] == row['nurse_id']) &
            (pref_df['date'] == row['date']) &
            (pref_df['shift'] == row['shift']) &
            (pref_df['ward'] == row['ward'])).any()

candidate_df['is_preferred'] = candidate_df.apply(is_preferred, axis=1)
candidate_df['label'] = 0  # All are unassigned (candidate edges)

test_assigned = candidate_df
print(f"Generated {test_assigned.shape[0]} candidate nurse-shift edges for live scheduling.")

# --- Step 3: Graph construction (saves nurse/shift maps) ---
train_graph = build_graph(train_assigned, save_mapping_dir=DATA_DIR)
test_graph = build_graph(test_assigned, save_mapping_dir=DATA_DIR)

# --- Step 4: Train model ---
gat_model = train_gat(
    train_graph,
    in_dim=train_graph.x.shape[1],
    hidden_dim=cfg['model']['gat_hidden_dim'],
    out_dim=1,
    heads=cfg['model']['gat_heads'],
    epochs=cfg['model']['gat_epochs'],
    lr=cfg['model']['gat_lr']
)

# --- Step 5: Predict on all candidates for live window ---
gat_scores = predict_gat(gat_model, test_graph)
print("Type of gat_scores:", type(gat_scores))
print("Shape of gat_scores:", getattr(gat_scores, "shape", "No shape attr"))
print("gat_scores example:", gat_scores[:10])

# --- Step 6: Export edge scores ---
edges = test_graph.edge_index.cpu().numpy().T
df_edges = pd.DataFrame(edges, columns=['nurse_node', 'shift_node'])
df_edges['gat_score'] = gat_scores
edges_csv = os.path.join(DATA_DIR, "edges_with_scores.csv")
df_edges.to_csv(edges_csv, index=False)
print(f"Exported assignment scores to {edges_csv}")

# --- Step 7: Postprocess for assignment solver ---
edges_readable_path = prepare_edges_for_assignment(
    edges_path=edges_csv,
    nurse_map_path=os.path.join(DATA_DIR, "nurse_idx_map.csv"),
    shift_map_path=os.path.join(DATA_DIR, "shift_idx_map.csv"),
    output_path=os.path.join(DATA_DIR, "edges_with_scores_readable.csv")
)
print(f"Readable edge list for assignment: {edges_readable_path}")

# --- Step 8: Assignment solver ---
final_assignment_path = solve_assignment(
    edges_with_scores_path=edges_readable_path,
    output_assignment_path=os.path.join(DATA_DIR, "assignment_ui_output.csv")
)
print(f"Final UI assignment output: {final_assignment_path}")