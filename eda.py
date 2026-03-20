 #Phase 1: Load & Explore the Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"E:\projects\Customer_Segmentation_Behavior_Analysis\online_retail.csv", encoding='ISO-8859-1')


print(df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())


df = df.dropna(subset=['CustomerID'])           # remove rows without customer
df = df[df['Quantity'] > 0]                     # remove returns/cancellations
df = df[df['UnitPrice'] > 0]                    # remove zero-price rows
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
top_countries.plot(kind='bar', title='Top 10 Countries by Revenue', figsize=(10,5))
plt.tight_layout()
plt.savefig('top_countries.png')
plt.show()


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Month'] = df['InvoiceDate'].dt.to_period('M')
monthly = df.groupby('Month')['TotalPrice'].sum()
monthly.plot(title='Monthly Revenue Trend', figsize=(10,5))
plt.tight_layout()
plt.savefig('monthly_revenue.png')
plt.show()

print("Clean dataset shape:", df.shape)
print("Unique customers:", df['CustomerID'].nunique())