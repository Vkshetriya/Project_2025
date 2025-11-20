#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-20T15:26:37.203Z
"""

# ## <b> Sales Forecasting Using Random Forest
# 


# ## <b>Importing libraries


import pandas as pd
import matplotlib.pyplot as plt


# ## <b> Data Loading & Preprocessing
# 


df = pd.read_csv(r"D:\dekstop\big data analytics\archive\Online_Retail_Data_Set.csv",encoding = "latin1")


df

#Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')


df['InvoiceDate']

df.head()

df.describe()

df.info()

df[['InvoiceNo']].head()


#count how  many orders are cancelled
df[df['InvoiceNo'].astype(str).str.startswith('C')]['InvoiceNo'].head()


#Remove cancelled orders
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]


df[df['InvoiceNo'].astype(str).str.startswith('C')]['InvoiceNo'].head()


df[df['Quantity'] == 0].shape


#check rows with missing CustomerID
df['CustomerID'].isna().sum()


#Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])


df['CustomerID'].isna().sum()

# Check & Count duplicates
df.duplicated().sum()


#Remove Duplicate Values
df = df.drop_duplicates()


df.duplicated().sum()

#Check for negative or zero UnitPrice
df[df['UnitPrice'] <= 0].shape


#Remove negative Unitprice
df = df[df['UnitPrice'] > 0]


df[df['UnitPrice'] <= 0].shape


df.isnull().sum()


# Create revenue column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


df.head(5)

# ## <b> Feature Extraction


#Extract date features (for seasonality analysis)
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()


df.head()

df['Year'].unique()


df.dtypes


# ## <b> Revenue Aggregation & EDA


# Top revenue StockCodes
stockcode_revenue = (
    df.groupby('StockCode')['TotalPrice']
      .sum()
      .sort_values(ascending=False)
)

stockcode_revenue

# ## Interpretation
# These stock codes represent the most financially important items in the inventory.
# 
# They contribute significantly more revenue compared to most other products.
# 
# These items likely have high sales volume, high price, or both.


print("Top 10 StockCodes by Revenue:")
top10_stockcodes = stockcode_revenue.head(10)
print(stockcode_revenue.head(10))

colors = plt.cm.Blues(range(50, 250, 20))  # gradient shades
plt.figure(figsize=(10,6))
plt.bar(top10_stockcodes.index, top10_stockcodes.values, color=colors[:10])
plt.xticks(rotation=75)
plt.ylabel("Revenue (INR)")
plt.title("Top 10 StockCodes by Revenue")
plt.tight_layout()
plt.show()


# ## Interpretation:
# This bar chart ranks the top 10 StockCodes (product IDs) based on their total revenue.
# A small group of StockCodes contributes disproportionately more revenue than the rest.
# The highest revenue StockCode crosses ₹1.6 lakh, indicating strong customer demand.
# Revenue steadily declines from left to right, showing that top performers significantly outperform the rest.
# These StockCodes are your bestselling SKUs and should be prioritized in inventory, marketing, and forecasting.
# Business Insight:
# Strong dependence on these 10 SKUs could affect revenue if any one goes out of stock.
# Maintaining supply continuity for these products is essential.


#Top revenue Product Descriptions
product_revenue = (
    df.groupby('Description')['TotalPrice']
      .sum()
      .sort_values(ascending=False)
)

print("\nTop 10 Products (Descriptions) by Revenue:")
print(product_revenue.head(10))


import matplotlib.pyplot as plt
top10_products = product_revenue.head(10)
plt.figure(figsize=(10,6))
plt.bar(top10_products.index, top10_products.values)
plt.xticks(rotation=75)
plt.ylabel("Revenue (INR)")
plt.title("Top 10 Products by Revenue")
plt.tight_layout()
plt.show()

# ## Interpretation
# This chart lists the actual product names contributing the highest revenue.
# The top product “PAPER CRAFT, LITTLE BIRDIE” alone makes ₹168,469.
# Other products like “REGENCY CAKESTAND 3 TIER” and “HEART T-LIGHT HOLDER” also perform strongly.
# These items appear to be decorative, gifting, and home décor products — a major revenue-driving category.
# Business Insight:
# Marketing campaigns and promotional bundles for these top items can directly increase revenue.
# These categories should be stocked heavily during festivals and seasonal peaks.


#Revenue Concentration (Top 10 Contribution)
total_revenue = df['TotalPrice'].sum()
top10_contribution = (product_revenue.head(10).sum() / total_revenue) * 100

print(f"\nTop 10 products contribute: {top10_contribution:.2f}% of total revenue")

#Pareto Analysis (80/20 rule)
product_revenue_percent = (product_revenue / total_revenue) * 100
cumulative_percent = product_revenue_percent.cumsum()

pareto_cutoff = cumulative_percent[cumulative_percent <= 80]


pareto_cutoff

# This part checks whether 20% of products generate 80% of revenue (Pareto rule).
# found 809 products contribute 80% of the revenue — meaning revenue is widely distributed across many products.
# Business Insight:
# The business is not dependent on just a few items, reducing risk.
# But forecasting becomes more complex because demand is spread over many SKUs.


#Top selling products:
product_sales = df.groupby('StockCode')['TotalPrice'].sum().sort_values(ascending=False)


print(f"\nNumber of products contributing to 80% revenue: {len(pareto_cutoff)}")
print("These are the top products following the Pareto Principle:")
print(pareto_cutoff)

#Dependency Risk — % revenue from single top product
top1_revenue_percent = (
    product_revenue.iloc[0] / total_revenue * 100
)


print(f"\nTop 1 product alone contributes: {top1_revenue_percent:.2f}%")


# The top product contributes only 1.9% of total revenue.
# This confirms that:
# No single item dominates revenue
# Loss of one product will not significantly affect total revenue.
# Product portfolio is diverse and stable.
# Business dependency risk is low
# Business Insight:
# 


# ## <b> Customer Segmentation (RFM Analysis)


df = df.dropna(subset=['CustomerID'])


df

#Calculate Recency, Frequency, Monetary (RFM)

# Set snapshot date (max date + 1 day)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# RFM calculation
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,   # Recency
    'InvoiceNo': 'nunique',                                    # Frequency
    'TotalPrice': 'sum'                                        # Monetary
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm.head()


#Create RFM Scores (1–4)
rfm['R_score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4])
rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1,2,3,4])




rfm['RFM_score'] = (
    rfm['R_score'].astype(int) +
    rfm['F_score'].astype(int) +
    rfm['M_score'].astype(int)
)


#Customer Segments
def segment_customer(row):
    if row['RFM_score'] >= 10:
        return 'High-Value Loyal'
    elif row['RFM_score'] >= 7:
        return 'Potential Loyal'
    elif row['RFM_score'] >= 5:
        return 'Price Sensitive'
    else:
        return 'One-time / Churned'

rfm['Segment'] = rfm.apply(segment_customer, axis=1)


rfm.head()


# ## <B> Time-Series Aggregation


# Daily revenue
daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum()
daily_sales.head()


# Weekly revenue
weekly_sales = df.groupby(df['InvoiceDate'].dt.to_period('W'))['TotalPrice'].sum()
weekly_sales.head()

# Monthly revenue
monthly_sales = df.groupby(df['InvoiceDate'].dt.to_period('M'))['TotalPrice'].sum()
monthly_sales.head()

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
monthly_sales.plot(kind='line')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()


#Identify Seasons (High vs Low Demand)
monthly_sales.sort_values(ascending=False).head()


monthly_sales.sort_values().head()


# ## Geographic Market Performance
# 


#Top Countries by Revenue
country_rev = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
country_rev


#Top 10 Countries
country_rev.head(10).plot(kind='bar', figsize=(12,5))
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")
plt.show()


# ## Interpretation
# Strong-Performing Region-United Kingdom
# Dominates the total revenue (over ~7.2 million)
# Represents the core market
# Indicates high brand awareness and strong customer loyalty
# Should be the primary focus for:
# ✔ Retention
# ✔ Premium product launches
# ✔ Subscription/loyalty programs
# Mid-Performing Regions (Growth Opportunity) - Netherlands, EIRE (Ireland), Germany, France, Australia
# Low-Performing Regions (Need Attention)-Spain, Switzerland, Belgium, Sweden, Canada, USA, Malta, UAE, Brazil, Saudi Arabia, etc.
# Hence,The UK is the strongest and most reliable market, but the business is heavily dependent on it. Mid-performing European markets offer good growth potential, while numerous low-performing markets need targeted marketing to expand customer reach. This segmentation helps allocate budget and strategy effectively.
# 


# ## Basket Size & Purchase Behavior


#Average Order Quantity per Invoice

# Basket size = average quantity per invoice
basket_size = df.groupby('InvoiceNo')['Quantity'].sum()

basket_size.mean()


def categorize_product(description):
    desc = str(description).lower()
    if "cup" in desc or "mug" in desc:
        return "Cups & Mugs"
    elif "light" in desc or "lantern" in desc:
        return "Lighting"
    elif "bag" in desc:
        return "Bags"
    elif "holder" in desc:
        return "Home Decor"
    elif "set" in desc:
        return "Gift Sets"
    elif "christmas" in desc:
        return "Christmas Items"
    else:
        return "Others"

df['Category'] = df['Description'].apply(categorize_product)


#Category-wise Revenue
category_revenue = df.groupby('Category')['TotalPrice'].sum().sort_values(ascending=False)
category_revenue


#Basket size across categories
category_basket = df.groupby('Category')['Quantity'].mean().sort_values(ascending=False)
category_basket


#Average Order Value (AOV)
AOV = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
AOV


import warnings
warnings.filterwarnings('ignore')


#AOV per Category
category_aov = df.groupby('Category').apply(
    lambda x: x.groupby('InvoiceNo')['TotalPrice'].sum().mean()
).sort_values(ascending=False)

category_aov


#Invoice-Level Summary
invoice_summary = df.groupby('InvoiceNo').agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum'
})

invoice_summary.head()


# These differences confirm that customer buying behavior varies widely—ranging from small occasional buyers to heavy bulk purchasers.


# ## Interpretation 
# Customers purchase in bulk → high basket average
# AOV is strong and largely driven by large-quantity buyers
# Home Decor and Lighting are top-performing categories
# Price has almost no effect on quantity
# Premium items sell in very small volumes
# Customer shopping patterns vary widely


# ## Price Sensitivity & Discount Strategy Impact


#Check relationship between UnitPrice and Quantity
# Price vs Quantity correlation
price_quantity_corr = df['UnitPrice'].corr(df['Quantity'])
price_quantity_corr


# The correlation between Unit Price and Quantity Purchased is -0.004 (almost zero).
# This shows that customers are not highly price-sensitive.
# Meaning: small price changes do NOT significantly impact how much customers buy.


#Categorize products into price groups
df['PriceGroup'] = pd.cut(df['UnitPrice'],
                          bins=[0, 2, 5, 20, df['UnitPrice'].max()],
                          labels=['Low', 'Medium', 'High', 'Premium'])


#average quantity per price group
df.groupby('PriceGroup')['Quantity'].mean()


# ## Interpretation
# Low-priced products generate maximum volume.
# Premium products move in very small quantities.
# Volume decreases consistently as price increases.
# For Low–Medium Priced Products:
# High volume potential → customers respond to bundles.
# Use combo offers, small discounts, and bulk incentives.
# Goal: maximize sales volume + AOV.
# For High–Premium Products:
# Low volume but higher revenue per unit.
# Customers focus on quality, not price.
# Avoid heavy discounts → may reduce brand value.
# Use value-based pricing and premium branding.


df_group = df.groupby('PriceGroup')['Quantity'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=df_group, x='PriceGroup', y='Quantity')

plt.title("Average Quantity Purchased by Price Segment")
plt.xlabel("Price Category")
plt.ylabel("Avg Quantity Bought")
plt.show()


# ## 
# This bar chart analyzes how product pricing affects buying behavior.
# divided products into four price segments — Low, Medium, High, Premium — and calculated how many units customers typically buy in each segment.
# -Low-priced products have the highest demand
# Customers buy ~18 units on average from the Low price category.
# These items are impulse purchases, everyday essentials, or low-risk items.
# Lower prices → higher volume.
# -Demand decreases as price increases
# Medium-priced items average 8–9 units → volume declines almost 50% from low-priced items.
# High-priced items drop further to ~4 units.
# Premium products have very low volume (around 2 units).
# This shows a clear negative price–quantity relationship:
# Higher price → Lower quantity purchased.


# ## <b> Random forest regression


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Convert to proper timestamp index
monthly_sales = monthly_sales.to_timestamp().to_frame(name="Revenue")
monthly_sales.head()


# ## <b> Feature Engineering


# Create time-series features
monthly_sales['Lag1'] = monthly_sales['Revenue'].shift(1)
monthly_sales['Lag2'] = monthly_sales['Revenue'].shift(2)
monthly_sales['Lag3'] = monthly_sales['Revenue'].shift(3)

monthly_sales['Roll3'] = monthly_sales['Revenue'].rolling(3).mean()
monthly_sales['Roll6'] = monthly_sales['Revenue'].rolling(6).mean()

monthly_sales['Diff'] = monthly_sales['Revenue'].diff()

monthly_sales = monthly_sales.dropna()
monthly_sales.head()


# ## <b> Train–Test Split & Build Feature Set


train = monthly_sales.iloc[:-3]
test = monthly_sales.iloc[-3:]

X_train = train.drop('Revenue', axis=1)
y_train = train['Revenue']

X_test = test.drop('Revenue', axis=1)
y_test = test['Revenue']


X = monthly_sales[['Lag1', 'Lag2', 'Lag3', 'Roll3', 'Roll6', 'Diff']]
y = monthly_sales['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ## <b> Train the Random Forest Model


model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# ## <b> Evaluate Model


pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("MAE:", mae)
print("RMSE:", rmse)
print("Actual:", list(y_test.values))
print("Predicted:", list(pred))


# ## <b> Rolling Forecast for Next 3 Months


last_values = monthly_sales.iloc[-1]   # most recent row

last_lag1 = last_values['Revenue']
last_lag2 = last_values['Lag1']
last_lag3 = last_values['Lag2']
last_roll3 = last_values['Roll3']
last_roll6 = last_values['Roll6']
last_diff = last_values['Diff']

future_predictions = []
future_dates = []

for i in range(1, 4):  # 3-month forecast
    
    next_date = monthly_sales.index[-1] + pd.DateOffset(months=i)
    future_dates.append(next_date)

    row = pd.DataFrame([[
        last_lag1, last_lag2, last_lag3,
        last_roll3, last_roll6, last_diff
    ]], columns=X.columns)

    pred_value = model.predict(row)[0]
    future_predictions.append(pred_value)

    # ---- UPDATE VALUES FOR NEXT MONTH ----
    last_diff = pred_value - last_lag1
    last_roll3 = (pred_value + last_lag1 + last_lag2) / 3
    last_roll6 = (last_roll6 * 6 - last_lag3 + pred_value) / 6

    last_lag3 = last_lag2
    last_lag2 = last_lag1
    last_lag1 = pred_value

# ---- Final Forecast Table ----
forecast_df = pd.DataFrame({
    "InvoiceDate": future_dates,
    "Predicted_Revenue": future_predictions
})

forecast_df

# ## Interpretation
# The forecasting model estimates the company’s revenue for the next three months as follows:
# January 2012: ₹ 855,945.92
# February 2012: ₹ 967,847.95
# March 2012: ₹ 906,766.56
# These predictions indicate a positive growth trend at the beginning of the year. Revenue is expected to increase sharply in February, reaching its highest level among the three months. This suggests stronger customer demand or seasonal effects during that period.
# Although revenue is projected to slightly decline in March, it still remains higher than January, showing sustained demand and overall stability.
# From a business perspective, this forecast helps guide operational planning.Companies should prepare for higher stock requirements in January and especially February, ensuring that inventory levels are sufficient to meet the projected demand and avoid stockouts. For March, a moderate stocking strategy is recommended to prevent overstocking while still supporting strong sales levels.
# Overall, the forecast supports a healthy revenue outlook and provides useful direction for budgeting, procurement, and inventory management for the upcoming quarter.