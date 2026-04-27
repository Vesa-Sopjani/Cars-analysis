def basic_statistics(df):
    print(df.describe())
    print(df.median(numeric_only=True))
    print(df.mode().iloc[0])