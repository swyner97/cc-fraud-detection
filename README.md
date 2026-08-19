# Credit Card Fraud Detection

A data analysis project for exploring patterns in fraudulent credit card transactions using exploratory data analysis and interactive Power BI visualizations.

## Project Overview

This project analyzes a dataset of 1.3M credit card transactions to identify patterns in fraudulent activity. The dataset contains 7.51K fraudulent transactions (0.58% fraud rate) across diverse merchant categories, geographic regions, and customer demographics. The analysis combines exploratory data analysis (EDA) with interactive Power BI dashboards for insights and visualization.

### Key Metrics
- **Total Transactions**: 1.30M
- **Fraud Cases**: 7.51K
- **Overall Fraud Rate**: 0.58%
- **Total Transaction Amount**: $91.22M
- **Total Fraud Amount**: $3.99M

## Key Findings

### High-Risk Indicators (Fraud Rate Multipliers)
- **Night Transactions**: 14.3x increase in fraud likelihood
- **High Transaction Velocity**: 3.1x increase (4+ purchases in 6 hours)
- **Large Amounts**: 3.5x increase when amount exceeds $118.30
- **Online Shopping**: 3.8x increase
- **In-Store Grocery**: 3.0x increase

### Demographics & Geography
- **Age Risk**: Fraud rate increases with age, peaking at 65+ (0.75%)
- **Gender Risk**: Males show slightly higher fraud rate (0.64%) vs females (0.52%)
- **Urban Risk**: Urban areas have 1.4x higher fraud rate (0.67%) vs rural (0.57%)

### Geographic Hotspots (Among states with 500+ transactions)
1. **Rhode Island**: 2.73% fraud rate
2. **Colorado**: 2.30% fraud rate
3. **Oregon**: 2.15% fraud rate
4. **Tennessee**: 2.10% fraud rate

### Merchant Category Risk
- **Personal Care**: 1.76% fraud rate (highest)
- **Shopping Net**: 1.74% fraud rate
- **Grocery POS**: 1.40% fraud rate
- **Misc Net**: 1.43% fraud rate

### Temporal Patterns
- **Late Night Peak**: 12 AM–3 AM shows 1.3%+ fraud rate (weekend peaks ~1.4%)
- **Day of Week**: Saturday has highest late-night fraud concentration
- **Business Hours**: Daytime transactions (7 AM–11 PM) show lower fraud rates (<0.2%)

## Project Structure

```
cc-fraud-detection/
├── data/
│   ├── fraudTrain.csv              # Raw transaction data (1.3M transactions)
│   └── processed/
│       ├── powerbi_transactions.csv # Cleaned transactions for Power BI
│       ├── summary_by_age_group.csv
│       ├── summary_by_amt_bracket.csv
│       ├── summary_by_category.csv
│       ├── summary_by_gender.csv
│       ├── summary_by_hour.csv
│       ├── summary_by_location_type.csv
│       └── summary_by_state.csv
│
├── img/
│   ├── class_distribution.png
│   ├── fraud_by_age.png
│   ├── fraud_by_amount.png
│   ├── fraud_by_gender.png
│   ├── fraud_by_location.png
│   ├── fraud_by_merchant_category.png
│   ├── transaction_amount_histogram.png
│   └── transaction_time_distribution.png
│
├── notebooks/
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   └── Fraud_Detection.ipynb       # Additional exploration notebook
│
├── src/
│   ├── __pycache__/
│   ├── clean_data.py
│   ├── features.py
│   ├── transformations.py
│   └── .gitignore
│
├── powerbi_cc.pdf                  # Interactive Power BI dashboard (backup)
├── powerbi_prep.py                 # Data preparation for Power BI export
├── README.md                       # Project documentation
├── main.py                         # Main execution script
└── .gitignore                      # Git ignore file
```

## Technology Stack

- **Python 3.8+**
- **pandas** – Data manipulation and aggregation
- **NumPy** – Numerical computing
- **Matplotlib / Seaborn** – Static visualizations
- **Power BI** – Interactive dashboards and drill-downs

## Notebooks

### `eda.ipynb` – Exploratory Data Analysis
Comprehensive analysis of transaction patterns:
- Data cleaning and feature engineering (age groups, amount brackets, location types)
- Statistical summaries by demographic and geographic segments
- Visualization of fraud distribution across key dimensions
- Temporal pattern analysis

**Key Outputs**: Summary tables for Power BI, static visualizations

### `Fraud_Detection.ipynb`
Supporting notebook with additional analysis and data exploration.

## Setup & Usage

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn
```

### 2. Run EDA
```bash
jupyter notebook eda.ipynb
```
Generates exploratory visualizations and builds summary tables for Power BI.

### 3. Prepare Data for Power BI
```bash
python powerbi_prep.py
```

**Input**: `data/fraudTrain.csv`

**Outputs to `data/processed/`**:
- `powerbi_transactions.csv` – Full transaction table with engineered features
- `summary_by_*.csv` – Aggregated fraud rates by dimension

**Key Features Engineered**:
- `age_group` – Bucketed into: 18-24, 25-34, 35-44, 45-54, 55-64, 65+
- `amt_bracket` – Bucketed into: $0-$50, $50-$100, $100-$500, $500-$1K, $1K-$5K, $5K+
- `location_type` – Categorized into: Rural, Suburban, Urban

**PII Removed**: Social Security numbers, credit card numbers, names, exact addresses, precise coordinates

### 4. Explore Interactive Dashboard
Open `powerbi_cc.pdf` in Power BI Desktop

## Power BI Dashboard

### Dashboard Overview

The dashboard includes four interactive pages:

### Page 1: Key Influencers
- Identifies what drives fraud status using Power BI's Key Influencers visual
- Highlights the strongest predictors (Night Transaction, Online Shopping, High Velocity)
- Shows fraud likelihood multipliers for each factor

### Page 2: Demographic & Geographic Analysis
- **Fraud by Age Group**: Stacked bar chart with transaction volume and fraud count
- **Fraud Transaction Count by State**: Choropleth map showing regional hotspots
- **Fraud by Time & Day**: Heatmap of fraud rates across hours and days of week
- **Filters**: Age Group, Gender, Weekend Status, Night Transaction, State

### Page 3: Category & Amount Analysis
- **Top States by Fraud Rate**: Horizontal bar chart ranking states
- **Top Categories by Fraud Rate**: Online Shopping, Online Misc, In-Store Grocery
- **Fraud Volume vs Rate**: Bubble chart showing state risk (fraud rate) vs. scale (transaction volume)
- **Insights Panel**: Actionable recommendations based on current selections

### Page 4: Detailed Insights & Recommendations
Summary of findings with strategic recommendations for fraud monitoring and prevention.

**Slicers Available**: State, Category, Gender, Week, Amount Bracket

## Key Insights & Recommendations

### Immediate Actions
1. **Enhanced Night Monitoring**: Implement real-time fraud checks for transactions between 12 AM–5 AM (14.3x risk multiplier)
2. **High-Velocity Detection**: Flag accounts with 4+ transactions in 6 hours (3.1x risk)
3. **Geographic Focusing**: Prioritize fraud review in RI, CO, OR, and TN
4. **Category Deep-Dive**: Focus on Online Shopping, Personal Care, and Shopping Net categories

### Data Quality & Monitoring
- Regularly validate feature distributions (age, amount, location)
- Monitor for temporal pattern shifts
- Update age brackets and amount bins as customer behavior evolves

## Data Preparation Details

The `powerbi_prep.py` script:

1. **Feature Engineering**
   ```python
   df["age_group"] = pd.cut(df["age"], bins=[18, 25, 35, 45, 55, 65, 100], 
                            labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"])
   df["amt_bracket"] = pd.cut(df["amt"], bins=[0, 50, 100, 500, 1000, 5000, 30000],
                              labels=["$0-$50", "$50-$100", "$100-$500", "$500-$1K", "$1K-$5K", "$5K+"])
   df["location_type"] = pd.cut(df["city_pop"], bins=[0, 25000, 100000, float('inf')],
                                labels=["Rural", "Suburban", "Urban"])
   ```

2. **Privacy Protection**: Removes PII (credit card numbers, names, addresses, GPS coordinates, merchant exact locations)

3. **Fraud Rate Calculation**
   ```python
   fraud_rate_table = df.groupby(dimension)["is_fraud"].agg(
       total_transactions="count",
       fraud_count="sum"
   )
   fraud_rate_table["fraud_rate"] = fraud_rate_table["fraud_count"] / fraud_rate_table["total_transactions"]
   ```

4. **Export**: Saves transactions and summary tables as CSV for Power BI

## Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `trans_id` | int | Unique transaction identifier |
| `is_fraud` | binary | Target: 1 = Fraudulent, 0 = Legitimate |
| `amt` | float | Transaction amount ($) |
| `category` | string | Merchant category (e.g., "Online Shopping", "Gas/Transport") |
| `state` | string | Cardholder state |
| `age` | int | Cardholder age (years) |
| `gender` | string | Cardholder gender (M/F) |
| `city_pop` | int | Population of merchant city |
| `hour` | int | Hour of transaction (0–23) |
| `day_of_week` | int | Day of week (0=Monday, 6=Sunday) |
| `night_trans` | binary | 1 if transaction between 12 AM–5 AM |
| `age_group` | string | Engineered: binned age |
| `amt_bracket` | string | Engineered: binned amount |
| `location_type` | string | Engineered: Rural/Suburban/Urban |

**Last Updated**: August 2026  
**Dataset Size**: 1.30M transactions | **Fraud Cases**: 7.51K (0.58%)  
**Analysis Coverage**: Multi-dimensional (demographic, geographic, temporal, categorical)