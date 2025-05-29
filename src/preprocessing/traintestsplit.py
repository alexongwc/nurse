import pandas as pd

def split_schedule_and_preferences(df, test_weeks=2, date_col='date'):
    """
    Returns:
        train_assigned_df: label==1, for model training (historical schedule, up to cutoff)
        test_assigned_df: label==1, for evaluation (historical schedule, after cutoff)
        train_pref_df: label==0, for candidate preferences overlapping train period
        test_pref_df: label==0, for candidate preferences overlapping test period
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Assigned shifts (label==1)
    assigned_df = df[df['label'] == 1]
    # Preferences (label==0)
    pref_df = df[df['label'] == 0]
    
    # Find test cutoff by weeks on ASSIGNED shifts
    max_date = assigned_df[date_col].max()
    cutoff_date = max_date - pd.Timedelta(days=7*test_weeks - 1)
    
    # Split assigned edges (true schedule)
    train_assigned_df = assigned_df[assigned_df[date_col] < cutoff_date].reset_index(drop=True)
    test_assigned_df  = assigned_df[assigned_df[date_col] >= cutoff_date].reset_index(drop=True)
    
    # Split preference edges by date as well (for model inference)
    train_pref_df = pref_df[pref_df[date_col] < cutoff_date].reset_index(drop=True)
    test_pref_df  = pref_df[pref_df[date_col] >= cutoff_date].reset_index(drop=True)
    
    print(f"Train assignments: {train_assigned_df.shape[0]} rows (label==1)")
    print(f"Test assignments: {test_assigned_df.shape[0]} rows (label==1)")
    print(f"Train preferences: {train_pref_df.shape[0]} rows (label==0)")
    print(f"Test preferences: {test_pref_df.shape[0]} rows (label==0)")
    print(f"Train last date: {train_assigned_df[date_col].max().date() if not train_assigned_df.empty else 'N/A'}")
    print(f"Test first date: {test_assigned_df[date_col].min().date() if not test_assigned_df.empty else 'N/A'}")

    return train_assigned_df, test_assigned_df, train_pref_df, test_pref_df
