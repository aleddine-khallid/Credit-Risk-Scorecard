
def filter_and_split(df):
    """
    Filters the dataset to only include records with GB_Flag in ['G', 'B']
    and returns development and validation samples based on the 'split' column.
    """
    df_model = df[df['GB_Flag'].isin(['Good', 'Bad'])].copy()
    print(f" Modeling data shape (only G/B): {df_model.shape}")

    dev = df_model[df_model['split'] == 'Development'].copy()
    val = df_model[df_model['split'] == 'Validation'].copy()

    print(f" Development sample shape: {dev.shape}")
    print(f" Validation sample shape: {val.shape}")

    return dev, val
