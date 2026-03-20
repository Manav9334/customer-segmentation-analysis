import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base_dir, "rfm_clustered.csv"))

rfm = load_data()

# Sidebar
st.sidebar.title("Filters")
selected_segments = st.sidebar.multiselect(
    "Select Segments",
    options=rfm['Cluster_Label'].unique(),
    default=rfm['Cluster_Label'].unique()
)
recency_range = st.sidebar.slider(
    "Recency Range (days)",
    int(rfm['Recency'].min()),
    int(rfm['Recency'].max()),
    (int(rfm['Recency'].min()), int(rfm['Recency'].max()))
)

filtered = rfm[
    (rfm['Cluster_Label'].isin(selected_segments)) &
    (rfm['Recency'] >= recency_range[0]) &
    (rfm['Recency'] <= recency_range[1])
]

# Title
st.title("Customer Segmentation Dashboard")
st.markdown("**RFM Analysis + K-Means Clustering** | UCI Online Retail Dataset")
st.divider()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",    f"{len(filtered):,}")
col2.metric("Total Revenue",      f"£{filtered['Monetary'].sum():,.0f}")
col3.metric("Avg Order Value",    f"£{filtered['Monetary'].mean():,.0f}")
col4.metric("Avg Recency (days)", f"{filtered['Recency'].mean():.0f}")
st.divider()

# Row 1
col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Distribution")
    seg_counts = filtered['Cluster_Label'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig1 = px.pie(seg_counts, values='Count', names='Segment',
                  color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
    fig1.update_layout(showlegend=True, margin=dict(t=20,b=20))
    st.plotly_chart(fig1)

with col2:
    st.subheader("Revenue by Segment")
    seg_rev = filtered.groupby('Cluster_Label')['Monetary'].sum().reset_index()
    seg_rev.columns = ['Segment', 'Revenue']
    seg_rev = seg_rev.sort_values('Revenue', ascending=True)
    fig2 = px.bar(seg_rev, x='Revenue', y='Segment', orientation='h',
                  color='Segment',
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  text=seg_rev['Revenue'].apply(lambda x: f'£{x:,.0f}'))
    fig2.update_traces(textposition='outside')
    fig2.update_layout(showlegend=False, margin=dict(t=20,b=20))
    st.plotly_chart(fig2)

# Row 2
col1, col2 = st.columns(2)
with col1:
    st.subheader("Recency vs Monetary")
    fig3 = px.scatter(filtered, x='Recency', y='Monetary',
                      color='Cluster_Label', size='Frequency',
                      hover_data=['CustomerID','Frequency'],
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      opacity=0.6)
    fig3.update_layout(margin=dict(t=20,b=20))
    st.plotly_chart(fig3)

with col2:
    st.subheader("Monetary Distribution")
    fig4 = px.box(filtered, x='Cluster_Label', y='Monetary',
                  color='Cluster_Label',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig4.update_layout(showlegend=False, margin=dict(t=20,b=20))
    st.plotly_chart(fig4)

# Heatmap
st.subheader("Average RFM Profile by Segment")
rfm_avg  = filtered.groupby('Cluster_Label')[['Recency','Frequency','Monetary']].mean().round(1)
rfm_norm = (rfm_avg - rfm_avg.min()) / (rfm_avg.max() - rfm_avg.min())
fig5 = go.Figure(data=go.Heatmap(
    z=rfm_norm.values,
    x=['Recency','Frequency','Monetary'],
    y=rfm_norm.index.tolist(),
    colorscale='RdYlGn', reversescale=True,
    text=rfm_avg.values, texttemplate="%{text}", textfont={"size":13}
))
fig5.update_layout(margin=dict(t=20,b=20), height=250)
st.plotly_chart(fig5)

# PCA
st.subheader("Cluster View (PCA)")
fig6 = px.scatter(filtered, x='PCA1', y='PCA2',
                  color='Cluster_Label',
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  opacity=0.5, hover_data=['CustomerID','Monetary'])
fig6.update_layout(margin=dict(t=20,b=20))
st.plotly_chart(fig6)

# Summary table
st.subheader("Segment Summary & Recommendations")
summary = filtered.groupby('Cluster_Label').agg(
    Customers     = ('CustomerID', 'count'),
    Avg_Recency   = ('Recency',    'mean'),
    Avg_Frequency = ('Frequency',  'mean'),
    Total_Revenue = ('Monetary',   'sum')
).round(1).reset_index()

recs = {
    'High Value':      'Reward with loyalty programs, ask for referrals',
    'Promising':       'Upsell with targeted offers, nurture engagement',
    'Needs Attention': 'Send win-back campaigns with discounts',
    'Lost':            'Low priority — survey to understand churn reason'
}
summary['Recommendation'] = summary['Cluster_Label'].map(recs)
st.dataframe(summary, hide_index=True)

st.divider()
st.markdown("© 2024 Customer Segmentation Analysis | Data from UCI Online Retail Dataset")