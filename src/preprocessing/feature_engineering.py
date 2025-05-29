# src/dataprep/feature_engineering.py

import pandas as pd
import numpy as np

def add_total_hours_in_fortnight(df, assigned_df):
    """Add feature: Total hours worked in last 2 weeks."""
    def get_total_hours(nurse_id, shift_date):
        mask = (assigned_df['nurse_id'] == nurse_id) & \
               (assigned_df['date'] < shift_date) & \
               (assigned_df['date'] >= shift_date - pd.Timedelta(days=13))
        return assigned_df.loc[mask, 'duration_hours'].sum() if mask.any() else 0

    df['hours_last_2wks'] = df.apply(
        lambda row: get_total_hours(row['nurse_id'], row['date']),
        axis=1
    )
    return df

def add_days_since_last_shift(df, assigned_df):
    """Add feature: Days since last assigned shift."""
    def days_since(nurse_id, shift_date):
        prev = assigned_df[(assigned_df['nurse_id'] == nurse_id) &
                           (assigned_df['date'] < shift_date)]
        if not prev.empty:
            last_date = prev['date'].max()
            return (shift_date - last_date).days
        else:
            return np.nan  # or a default value

    df['days_since_last_shift'] = df.apply(
        lambda row: days_since(row['nurse_id'], row['date']),
        axis=1
    )
    return df

def add_consecutive_work_days(df, assigned_df):
    """Add feature: Number of consecutive working days up to this shift."""
    def consecutive_days(nurse_id, shift_date):
        prev_days = assigned_df[(assigned_df['nurse_id'] == nurse_id) &
                                (assigned_df['date'] < shift_date)]
        if prev_days.empty:
            return 0
        dates = prev_days['date'].sort_values(ascending=False)
        count = 0
        last_date = shift_date - pd.Timedelta(days=1)
        for d in dates:
            if (last_date - d).days == 0:
                count += 1
                last_date -= pd.Timedelta(days=1)
            else:
                break
        return count

    df['consecutive_work_days'] = df.apply(
        lambda row: consecutive_days(row['nurse_id'], row['date']),
        axis=1
    )
    return df

def add_shift_and_ward_features(df):
    """Add features: day_of_week, one-hot shift, one-hot ward."""
    df['day_of_week'] = df['date'].dt.dayofweek
    shift_onehot = pd.get_dummies(df['shift'], prefix='shift')
    ward_onehot = pd.get_dummies(df['ward'], prefix='ward')
    df = pd.concat([df, shift_onehot, ward_onehot], axis=1)
    return df

def feature_engineering(edge_df):
    """
    Main entry: Add all features to edge_df.
    Assumes edge_df['date'] is already pd.Timestamp and that 'label' column exists.
    """
    # Only use label==1 for historical assignment-based features
    assigned_df = edge_df[edge_df['label'] == 1].copy()

    # Apply each feature function
    edge_df = add_total_hours_in_fortnight(edge_df, assigned_df)
    edge_df = add_days_since_last_shift(edge_df, assigned_df)
    edge_df = add_consecutive_work_days(edge_df, assigned_df)
    edge_df = add_shift_and_ward_features(edge_df)
    return edge_df