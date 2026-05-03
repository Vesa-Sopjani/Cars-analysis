def correlation_analysis(df):
    corr = df.corr(numeric_only=True)
    result = corr['price'].drop('price').sort_values(ascending=False)
    print(result)
    return result