# Phase 2: RFM Feature Engineering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


df = pd.read_csv("online_retail.csv", encoding='latin-1')

df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['TotalPrice']   = df['Quantity'] * df['UnitPrice']
df['InvoiceDate']  = pd.to_datetime(df['InvoiceDate'])
df['CustomerID']   = df['CustomerID'].astype(int)

#Build RFM table 
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)  # reference date

rfm = df.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),
    Frequency = ('InvoiceNo',    'nunique'),
    Monetary  = ('TotalPrice',   'sum')
).reset_index()

print(rfm.head(10))
print("\nRFM Stats:")
print(rfm.describe())

# Score each dimension 1–4 
rfm['R_Score'] = pd.qcut(rfm['Recency'],   q=4, labels=[4,3,2,1])  # lower recency = better
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1,2,3,4])
rfm['M_Score'] = pd.qcut(rfm['Monetary'],  q=4, labels=[1,2,3,4])

rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# Label customer segments based on RFM scores 
def segment_customer(row):
    r = int(row['R_Score'])
    f = int(row['F_Score'])
    m = int(row['M_Score'])
    score = r + f + m

    if r == 4 and f == 4 and m == 4:
        return 'Champion'
    elif r >= 3 and f >= 3:
        return 'Loyal Customer'
    elif r == 4 and f <= 2:
        return 'New Customer'
    elif r >= 3 and f <= 2 and m <= 2:
        return 'Potential Loyalist'
    elif r == 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2 and m >= 3:
        return 'Big Spender (Inactive)'
    elif r == 1 and f >= 2:
        return 'Lost Customer'
    else:
        return 'Hibernating'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)

print("\nSegment Distribution:")
print(rfm['Segment'].value_counts())

# Visualizations 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RFM Analysis — Customer Segmentation', fontsize=16, fontweight='bold')

# Plot 1: Segment distribution
seg_counts = rfm['Segment'].value_counts()
axes[0,0].bar(seg_counts.index, seg_counts.values, color='steelblue')
axes[0,0].set_title('Customer Segments')
axes[0,0].set_xlabel('Segment')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=45)

# Plot 2: Revenue by segment
seg_revenue = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False)
axes[0,1].bar(seg_revenue.index, seg_revenue.values, color='coral')
axes[0,1].set_title('Revenue by Segment')
axes[0,1].set_xlabel('Segment')
axes[0,1].set_ylabel('Total Revenue (£)')
axes[0,1].tick_params(axis='x', rotation=45)

# Plot 3: RFM Scatter — Recency vs Monetary
scatter = axes[1,0].scatter(rfm['Recency'], rfm['Monetary'],
                             c=rfm['F_Score'].astype(int),
                             cmap='RdYlGn', alpha=0.5, s=20)
axes[1,0].set_title('Recency vs Monetary (color = Frequency)')
axes[1,0].set_xlabel('Recency (days)')
axes[1,0].set_ylabel('Monetary (£)')
plt.colorbar(scatter, ax=axes[1,0], label='Frequency Score')

# Plot 4: Avg RFM values per segment
seg_avg = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
seg_avg_norm = (seg_avg - seg_avg.min()) / (seg_avg.max() - seg_avg.min())  # normalize
seg_avg_norm.T.plot(kind='bar', ax=axes[1,1], colormap='tab10')
axes[1,1].set_title('Normalized Avg RFM by Segment')
axes[1,1].set_xlabel('RFM Dimension')
axes[1,1].set_ylabel('Normalized Score')
axes[1,1].tick_params(axis='x', rotation=0)
axes[1,1].legend(loc='upper right', fontsize=7)

plt.tight_layout()
plt.savefig('rfm_analysis.png', dpi=150)
plt.show()


rfm.to_csv('rfm_output.csv', index=False)
print("\nSaved rfm_output.csv — ready for Phase 3 clustering!")