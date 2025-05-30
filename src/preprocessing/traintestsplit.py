# src/preprocessing/traintestsplit.py

import pandas as pd

def split_schedule_and_preferences(
    df, 
    test_dates=None, 
    date_col='date'
):
    """
    If test_dates is provided (e.g., from preference.csv), use those for test split.
    Otherwise, use default week-based split (for evaluation).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    assigned_df = df[df['label'] == 1]
    pref_df = df[df['label'] == 0]

    if test_dates is not None:
        # Test set is exactly those dates
        test_dates = pd.to_datetime(test_dates)
        train_assigned_df = assigned_df[~assigned_df[date_col].isin(test_dates)].reset_index(drop=True)
        test_assigned_df = assigned_df[assigned_df[date_col].isin(test_dates)].reset_index(drop=True)
        train_pref_df = pref_df[~pref_df[date_col].isin(test_dates)].reset_index(drop=True)
        test_pref_df = pref_df[pref_df[date_col].isin(test_dates)].reset_index(drop=True)
    else:
        # Fallback: old logic (by last N weeks)
        max_date = assigned_df[date_col].max()
        cutoff_date = max_date - pd.Timedelta(days=7*2 - 1)  # Default 2 weeks
        train_assigned_df = assigned_df[assigned_df[date_col] < cutoff_date].reset_index(drop=True)
        test_assigned_df  = assigned_df[assigned_df[date_col] >= cutoff_date].reset_index(drop=True)
        train_pref_df = pref_df[pref_df[date_col] < cutoff_date].reset_index(drop=True)
        test_pref_df  = pref_df[pref_df[date_col] >= cutoff_date].reset_index(drop=True)

    print(f"Train assignments: {train_assigned_df.shape[0]} rows (label==1)")
    print(f"Test assignments: {test_assigned_df.shape[0]} rows (label==1)")
    print(f"Train preferences: {train_pref_df.shape[0]} rows (label==0)")
    print(f"Test preferences: {test_pref_df.shape[0]} rows (label==0)")
    print(f"Train last date: {train_assigned_df[date_col].max().date() if not train_assigned_df.empty else 'N/A'}")
    print(f"Test first date: {test_assigned_df[date_col].min().date() if not test_assigned_df.empty else 'N/A'}")

    return train_assigned_df, test_assigned_df, train_pref_df, test_pref_df
