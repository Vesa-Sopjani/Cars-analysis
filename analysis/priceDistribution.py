def price_distribution(df):
    print("Min price:", df['price'].min())
    print("Max price:", df['price'].max())
    print("Mean price:", df['price'].mean())
    print("Median price:", df['price'].median())