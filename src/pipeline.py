import pandas as pd
import yaml
import os

from src.preprocessing.feature_engineering import feature_engineering
from src.preprocessing.traintestsplit import split_schedule_and_preferences
from src.preprocessing.graphconstruction import build_graph
from src.model.model import train_gat, predict_gat

from src.postprocessing.postprocessing import prepare_edges_for_assignment
from src.postprocessing.assignmentsolver import solve_assignment

# --- Load config ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = "data"

# --- Step 1: Data prep ---
df = pd.read_csv(cfg['paths']['combined_csv'])
df['date'] = pd.to_datetime(df['date'])
df = feature_engineering(df)

# --- Step 2: Train-test split ---
train_assigned, test_assigned, train_pref, test_pref = split_schedule_and_preferences(
    df, test_weeks=cfg['split']['test_weeks']
)

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

# --- Step 5: Predict test edges ---
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