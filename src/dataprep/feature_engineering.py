import pandas as pd
import numpy as np

# Load combined dataset
edge_df = pd.read_csv('../data/combined.csv')

# Make sure date is datetime
edge_df['date'] = pd.to_datetime(edge_df['date'])

# For per-nurse stats, you'll need the historical schedule (label==1)
assigned_df = edge_df[edge_df['label'] == 1].copy()

# --- Total hours worked in last 2 weeks ---
def get_total_hours_in_fortnight(nurse_id, shift_date):
    mask = (assigned_df['nurse_id'] == nurse_id) & \
           (assigned_df['date'] < shift_date) & \
           (assigned_df['date'] >= shift_date - pd.Timedelta(days=13))
    if mask.any():
        return assigned_df.loc[mask, 'duration_hours'].sum()
    return 0

edge_df['hours_last_2wks'] = edge_df.apply(
    lambda row: get_total_hours_in_fortnight(row['nurse_id'], row['date']),
    axis=1
)

# --- Days since last assigned shift ---
def get_days_since_last_shift(nurse_id, shift_date):
    prev_shifts = assigned_df[(assigned_df['nurse_id'] == nurse_id) & 
                              (assigned_df['date'] < shift_date)]
    if not prev_shifts.empty:
        last_date = prev_shifts['date'].max()
        return (shift_date - last_date).days
    else:
        return np.nan  # Or a default value, e.g., 999

edge_df['days_since_last_shift'] = edge_df.apply(
    lambda row: get_days_since_last_shift(row['nurse_id'], row['date']),
    axis=1
)

# --- Number of consecutive working days up to this shift ---
def get_consecutive_days(nurse_id, shift_date):
    prev_days = assigned_df[(assigned_df['nurse_id'] == nurse_id) & (assigned_df['date'] < shift_date)]
    if prev_days.empty:
        return 0
    # Sort by date
    dates = prev_days['date'].sort_values(ascending=False)
    # Start counting from the most recent day before this shift
    count = 0
    last_date = shift_date - pd.Timedelta(days=1)
    for d in dates:
        if (last_date - d).days == 0:
            count += 1
            last_date -= pd.Timedelta(days=1)
        else:
            break
    return count

edge_df['consecutive_work_days'] = edge_df.apply(
    lambda row: get_consecutive_days(row['nurse_id'], row['date']),
    axis=1
)

# --- Shift node features ---
# Day of week
edge_df['day_of_week'] = edge_df['date'].dt.dayofweek  # 0=Monday

# One-hot encode shift type
shift_onehot = pd.get_dummies(edge_df['shift'], prefix='shift')
edge_df = pd.concat([edge_df, shift_onehot], axis=1)

# One-hot encode ward
ward_onehot = pd.get_dummies(edge_df['ward'], prefix='ward')
edge_df = pd.concat([edge_df, ward_onehot], axis=1)

# Save the new engineered dataset
edge_df.to_csv('../data/edges_for_gat_features.csv', index=False)

print("Feature engineering complete. New file: edges_for_gat_features.csv")
print(edge_df.head())