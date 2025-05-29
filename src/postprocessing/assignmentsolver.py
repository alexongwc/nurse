import pandas as pd
from ortools.linear_solver import pywraplp
import numpy as np

def solve_assignment(
    edges_with_scores_path="data/edges_with_scores_readable.csv",
    output_assignment_path="data/assignment_ui_output.csv",
    hours_per_fortnight=10
):
    """
    OR-Tools nurse scheduling solver with ward/shift/flex and hour constraints.
    Returns path to UI-ready output.
    """
    # 1. Load the edge list (with nurse/shift/date/ward/etc.)
    df = pd.read_csv(edges_with_scores_path)

    # 2. Extract basic sets
    nurses = df['nurse_id'].unique().tolist()
    shifts = df.index.tolist()  # each row is a nurse-shift application

    # 3. OR-Tools setup
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = [solver.BoolVar(f"x_{i}") for i in shifts]  # x[i] means assign the shift in row i

    # 4. Objective: maximize total GAT score
    solver.Maximize(solver.Sum(x[i] * df.loc[i, 'gat_score'] for i in shifts))

    # 5. Nurse hour constraints: each nurse must work at least 80 hours every 2 weeks
    df['week'] = pd.to_datetime(df['date']).dt.isocalendar().week
    # Build "fortnight" blocks (each 2-week period)
    df['fortnight'] = (df['week'] - df['week'].min()) // 2
    nurse_fortnights = df.groupby(['nurse_id', 'fortnight']).groups

    for (nurse, fn), rows in nurse_fortnights.items():
        solver.Add(
            solver.Sum([x[i] * df.loc[i, 'duration_hours'] for i in rows]) >= hours_per_fortnight
        )

    # 6. Ward staffing constraints for every shift (point in time)
    # For each shift time and ward, staff required:
    ward_requirements = {'C': 4, 'B': 2, 'A': 1, 'ICU': 1}
    # Group by date, start_time, end_time, ward
    unique_shifts = df.groupby(['date', 'start_time', 'end_time', 'ward']).groups

    for key, idxs in unique_shifts.items():
        ward = key[3]
        if ward in ward_requirements:
            solver.Add(
                solver.Sum([x[i] for i in idxs]) == ward_requirements[ward]
            )

    # 7. (Optional) Each nurse can't overlap shifts (simplest form, no two shifts at same start time)
    # You can uncomment for more realistic constraint
    # for nurse in nurses:
    #     nurse_shifts = df[df['nurse_id'] == nurse]
    #     times = nurse_shifts.groupby(['date', 'start_time']).groups
    #     for key, idxs in times.items():
    #         solver.Add(solver.Sum([x[i] for i in idxs]) <= 1)

    # 8. Solve!
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        chosen = [i for i in shifts if x[i].solution_value() > 0.5]
        df_out = df.iloc[chosen].copy()
        df_out.to_csv(output_assignment_path, index=False)
        print(f"Assignment saved to {output_assignment_path}")
        print(df_out.head())
        return output_assignment_path
    else:
        print("No feasible solution found.")
        # Optionally save empty output
        pd.DataFrame().to_csv(output_assignment_path, index=False)
        return None

# ---- Optional: CLI usage ----
if __name__ == "__main__":
    solve_assignment()