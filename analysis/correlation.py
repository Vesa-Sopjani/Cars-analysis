def correlation_analysis(df):
    corr = df.corr(numeric_only=True)
    print(corr['price'].sort_values(ascending=False))