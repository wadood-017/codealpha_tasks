import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


df = pd.read_csv("Unemployment_Rate.csv")

# Strip extra spaces 
df.columns = [c.strip() for c in df.columns]
for col in df.select_dtypes(include=["object"]):
    df[col] = df[col].str.strip()

# Convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Convert numeric columns to numbers safely
num_cols = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)",
    "longitude",
    "latitude"
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Extract time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["MonthName"] = df["Date"].dt.strftime("%b")

# Simplify unemployment rate reference
df["UnempRate"] = df["Estimated Unemployment Rate (%)"]

# Weighted average function for national rate
def weighted_rate(group):
    rates = group["UnempRate"]
    weights = group["Estimated Employed"]
    if weights.sum() == 0 or np.isnan(weights.sum()):
        return rates.mean()
    return (rates * weights).sum() / weights.sum()

# Compute national unemployment rate
national = df.groupby("Date").apply(weighted_rate).rename("National_Unemployment_Rate").reset_index()
df = df.merge(national, on="Date", how="left")

zone_trends = df.groupby(["Date", "Region.1"])["UnempRate"].mean().reset_index()

# Monthly averages
monthly_avg = df.groupby("Month")["UnempRate"].mean().reindex(range(1, 13))

# Define time periods for COVID impact
pre_covid_mask = df["Date"] < pd.to_datetime("2020-03-01")
covid_mask = (df["Date"] >= pd.to_datetime("2020-03-01")) & (df["Date"] <= pd.to_datetime("2021-12-31"))
post_covid_mask = df["Date"] >= pd.to_datetime("2022-01-01")

pre_mean = df.loc[pre_covid_mask, "UnempRate"].mean()
covid_mean = df.loc[covid_mask, "UnempRate"].mean()
post_mean = df.loc[post_covid_mask, "UnempRate"].mean()

# Identify Covid peak
national_series = national.set_index("Date").sort_index()
peak_date = national_series["National_Unemployment_Rate"].idxmax()
peak_value = national_series["National_Unemployment_Rate"].max()

# Print results
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Pre-COVID Avg:", round(pre_mean, 2))
print("COVID Avg:", round(covid_mean, 2))
print("Post-COVID Avg:", round(post_mean, 2) if not np.isnan(post_mean) else "No data yet")
print("COVID Peak:", peak_date.date(), "with rate", round(peak_value, 2), "%")


plt.figure(figsize=(10,4))
plt.plot(national_series.index, national_series["National_Unemployment_Rate"], label="National Rate")
plt.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2021-12-31"), alpha=0.2, color="red", label="COVID Period")
plt.title("National Unemployment Rate Over Time")
plt.xlabel("Date"); plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()


zones = zone_trends["Region.1"].unique()
for z in zones:
    sub = zone_trends[zone_trends["Region.1"] == z].set_index("Date").sort_index()
    plt.figure(figsize=(10,3))
    plt.plot(sub.index, sub["UnempRate"])
    plt.axvspan(pd.to_datetime("2020-03-01"), pd.to_datetime("2021-12-31"), alpha=0.2, color="red")
    plt.title(f"Unemployment Rate Over Time - Region: {z}")
    plt.xlabel("Date"); plt.ylabel("Unemployment Rate (%)")
    plt.show()


plt.figure(figsize=(8,3))
plt.plot(monthly_avg.index, monthly_avg.values, marker="o")
plt.xticks(monthly_avg.index, [datetime(2000,m,1).strftime("%b") for m in monthly_avg.index])
plt.title("Average Unemployment Rate by Month")
plt.xlabel("Month"); plt.ylabel("Avg Unemployment Rate (%)")
plt.show()


plt.figure(figsize=(10,4))
box_data = [df[df["Month"]==m]["UnempRate"].dropna().values for m in range(1,13)]
plt.boxplot(box_data, labels=[datetime(2000,m,1).strftime("%b") for m in range(1,13)])
plt.title("Unemployment Rate Distribution by Month")
plt.xlabel("Month"); plt.ylabel("Unemployment Rate (%)")
plt.show()


plt.figure(figsize=(6,3))
plt.bar(["Pre-COVID","COVID","Post-COVID"],
        [pre_mean, covid_mean, 0 if np.isnan(post_mean) else post_mean],
        color=["skyblue","salmon","lightgreen"])
plt.title("Average Unemployment Rate: Pre vs COVID vs Post")
plt.ylabel("Unemployment Rate (%)")
plt.show()
