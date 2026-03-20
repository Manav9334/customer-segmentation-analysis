# Customer Segmentation & Behavior Analysis

An end-to-end project that analyzes e-commerce customer behavior using **RFM Analysis** and **K-Means Clustering**, with an interactive dashboard built in Streamlit.

---

## Live Demo

> Deploy your app and paste the link here:
> **[View Live Dashboard](https://customer-segmentation-analysis-manav.streamlit.app/)**

---

## Project Overview

This project segments 4,300+ customers from a real UK e-commerce dataset into distinct behavioral groups. By combining RFM (Recency, Frequency, Monetary) analysis with K-Means clustering, it helps businesses identify their most valuable customers and take targeted actions to maximize revenue.

**Key findings from the analysis:**
- Analyzed £8.5M in revenue across 4,300+ customers
- Identified 4 distinct customer segments using K-Means clustering
- Top 15% of customers (High Value segment) drive over 55% of total revenue
- PCA visualization explains 93.7% of variance in just 2 components

---

## Dashboard Preview

![Dashboard](dashboard_screenshot.png)

---

## Tech Stack

| Category | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (K-Means, PCA, StandardScaler) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Dataset | UCI Online Retail Dataset |

---

## Project Structure

```
customer-segmentation-analysis/
│
├── app.py                    # Streamlit dashboard
├── eda.py                    # Phase 1: Exploratory data analysis
├── rfm_analysis.py           # Phase 2: RFM feature engineering
├── kmeans_clustering.py      # Phase 3: K-Means clustering + PCA
│
├── OnlineRetail.csv          # Raw dataset (download separately)
├── rfm_output.csv            # RFM scores per customer
├── rfm_clustered.csv         # Final clustered dataset
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Manav9334/customer-segmentation-analysis.git
cd customer-segmentation-analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the **Online Retail dataset**  [Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail) and place `OnlineRetail.csv` in the project folder.

### 4. Run the pipeline in order
```bash
python eda.py                 # Phase 1: EDA
python rfm_analysis.py        # Phase 2: RFM Analysis
python kmeans_clustering.py   # Phase 3: Clustering
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Methodology

### Phase 1 — Exploratory Data Analysis
- Loaded and cleaned 500K+ transaction records
- Removed nulls, returns, and zero-price entries
- Visualized top countries by revenue and monthly sales trends

### Phase 2 — RFM Feature Engineering
RFM is an industry-standard customer segmentation technique used by Amazon, Flipkart, and every major e-commerce company.

| Metric | Definition | Business Meaning |
|---|---|---|
| **Recency** | Days since last purchase | How recently active |
| **Frequency** | Number of unique orders | How often they buy |
| **Monetary** | Total spend | How much they spend |

Each customer was scored 1–4 on each dimension and assigned to one of 8 behavioral segments.

### Phase 3 — K-Means Clustering
- Applied log transformation to reduce skew in Frequency and Monetary
- Standardized features using StandardScaler
- Used Elbow Method and Silhouette Score to determine optimal k=4
- Applied PCA for 2D visualization (93.7% variance explained)

### Phase 4 — Interactive Dashboard
Built a Streamlit dashboard with:
- KPI metrics (total customers, revenue, avg order value)
- Interactive filters by segment and recency
- Pie chart, bar chart, scatter plot, box plot, heatmap
- Business recommendations per segment

---

## Customer Segments

| Segment | Profile | Business Action |
|---|---|---|
| **High Value** | Recent, frequent, high spend | Reward with loyalty programs, ask for referrals |
| **Promising** | Moderate RFM, recently active | Upsell with targeted offers |
| **Needs Attention** | Declining activity | Win-back campaigns with discounts |
| **Lost** | Inactive, low spend | Survey to understand churn reason |

---

## Results

- **Elbow Method** suggested optimal k=4
- **Silhouette Score** of 0.38 at k=4 confirms well-separated clusters
- **PCA** reduced 3D RFM space to 2D retaining 93.7% variance
- Clear visual separation between all 4 customer clusters

---

## Dataset

**Online Retail Dataset** — UCI Machine Learning Repository  
- 541,909 transactions from a UK-based online retailer  
- Date range: December 2010 to December 2011  
- 4,338 unique customers after cleaning  
- Source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
streamlit
openpyxl
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Author

**Manav** — B.Tech CSE (Data Science)  
[GitHub](https://github.com/Manav9334)
---

