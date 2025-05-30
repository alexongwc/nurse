# src/preprocessing/metrics.py

import pandas as pd

def evaluate_preference_match(
    assignment_path="data/assignment_ui_output.csv",
    preference_path="data/preference.csv",
    output_path="data/preference_evaluation.csv"
):
    # Read your CSVs
    pred = pd.read_csv(assignment_path)
    pref = pd.read_csv(preference_path)

    # Standardize column names
    if 'preferred_shift' in pref.columns:
        pref = pref.rename(columns={'preferred_shift': 'shift_pref', 'preferred_ward': 'ward_pref'})
    else:
        pref = pref.rename(columns={'shift': 'shift_pref', 'ward': 'ward_pref'})

    # Merge on nurse_id, date, ward (most reliable for your setup)
    merged = pd.merge(
        pred, 
        pref, 
        left_on=['nurse_id', 'date', 'shift', 'ward'],
        right_on=['nurse_id', 'date', 'shift_pref', 'ward_pref'],
        how='left',
        indicator=True
    )

    # Preference match column: match if found in preference
    merged['preference_match'] = merged['_merge'] == 'both'

    # Overall match rate
    match_rate = merged['preference_match'].mean()
    print(f"Preference match rate: {match_rate:.2%}")

    # Per-nurse match rate
    nurse_pref_score = merged.groupby('nurse_id')['preference_match'].mean()
    print("\nPer nurse preference match rate:\n", nurse_pref_score)

    # Save for review
    merged.to_csv(output_path, index=False)
    print(f"\nDetailed match evaluation exported to: {output_path}")

    # Also return for programmatic use if needed
    return match_rate, nurse_pref_score

# Optional CLI usage
if __name__ == "__main__":
    evaluate_preference_match()
