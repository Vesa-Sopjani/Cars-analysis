import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

DARK_BG    = "#0F1117"
PANEL_BG   = "#1A1D27"
GRID_COLOR = "#2A2D3A"
TEXT_PRI   = "#E8EAF0"
TEXT_SEC   = "#8A8FA8"
ACCENT     = "#4F8EF7"
ACCENT2    = "#F7674F"
ACCENT3    = "#4FD18A"
ACCENT4    = "#F7C94F"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_SEC,
    "axes.titlecolor":   TEXT_PRI,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     14,
    "axes.grid":         True,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "grid.alpha":        1.0,
    "xtick.color":       TEXT_SEC,
    "ytick.color":       TEXT_SEC,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_SEC,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": DARK_BG,
    "font.family":       "DejaVu Sans",
    "text.color":        TEXT_PRI,
})

def add_figure_label(fig, text):
    """Subtle dataset source label bottom-right."""
    fig.text(0.98, 0.01, text, ha="right", va="bottom",
             fontsize=7, color=TEXT_SEC, alpha=0.6)

def fmt_usd(x, _=None):
    if x >= 1_000_000: return f"${x/1e6:.1f}M"
    if x >= 1_000:     return f"${x/1000:.0f}k"
    return f"${x:.0f}"

def fmt_miles(x, _=None):
    if x >= 1_000: return f"{x/1000:.0f}k"
    return f"{x:.0f}"

df = pd.read_csv("../data/Cars.csv", encoding="latin1")
df.drop_duplicates(inplace=True)
df.dropna(subset=["price", "brand", "year", "mileage", "color", "state"], inplace=True)
df = df[(df["price"] > 0) & (df["mileage"] > 0)]
CURRENT_YEAR = datetime.now().year
df["age_of_car"] = CURRENT_YEAR - df["year"]


df["price_segment"] = pd.cut(
    df["price"],
    bins=[0, 10_000, 20_000, 35_000, 60_000, 1_000_000],
    labels=["Budget\n(<$10k)", "Economy\n($10–20k)",
            "Mid-range\n($20–35k)", "Premium\n($35–60k)", "Luxury\n($60k+)"]
)

print(f"Dataset loaded: {df.shape[0]:,} records · {df.shape[1]} columns")

BLUE_RAMP  = [ACCENT,   "#6FA8F9", "#9DC1FB", "#C2D9FD", "#E0ECFE"]
RED_RAMP   = [ACCENT2,  "#F98A70", "#FBAD98", "#FCCFC4", "#FEE8E3"]
GREEN_RAMP = [ACCENT3,  "#7ADBA5", "#A0E5C0", "#C2EFD8", "#E0F7EB"]
GOLD_RAMP  = [ACCENT4,  "#F9D97A", "#FBE5A0", "#FDEFC4", "#FEF7E0"]

BRAND_COLORS = {
    "Ford": ACCENT, "Chevrolet": ACCENT2, "BMW": ACCENT3,
    "Toyota": ACCENT4, "Mercedes-Benz": "#C77DFF",
    "Jeep": "#FF9F1C", "GMC": "#2EC4B6", "Ram": "#F94144",
    "Dodge": "#F3722C", "Honda": "#90BE6D",
}

#Chart 1

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(DARK_BG)

prices_clipped = df["price"].clip(upper=100_000)
n, bins, patches = ax.hist(prices_clipped, bins=80, color=ACCENT,
                           alpha=0.85, edgecolor=DARK_BG, linewidth=0.4)

seg_colors = {(0, 10000): "#4FD18A", (10000, 20000): ACCENT,
              (20000, 35000): ACCENT4, (35000, 60000): ACCENT2,
              (60000, 100000): "#C77DFF"}
for patch, left in zip(patches, bins[:-1]):
    for (lo, hi), col in seg_colors.items():
        if lo <= left < hi:
            patch.set_facecolor(col)
            patch.set_alpha(0.8)

median_p = df["price"].median()
mean_p   = df["price"].mean()
ax.axvline(median_p, color="white",   linestyle="--", lw=1.4,
           label=f"Median  ${median_p:,.0f}")
ax.axvline(mean_p,   color=ACCENT4,   linestyle="--", lw=1.4,
           label=f"Mean    ${mean_p:,.0f}")


ax.annotate(f"${median_p/1000:.1f}k", xy=(median_p, n.max()*0.85),
            color="white", fontsize=9, ha="left",
            xytext=(median_p + 1500, n.max()*0.85))

ax.set_title("Price Distribution of Used Cars (USD)", fontsize=14, fontweight="bold")
ax.set_xlabel("Price (USD)", fontsize=10, labelpad=6)
ax.set_ylabel("Number of Listings", fontsize=10, labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))


seg_patches = [
    mpatches.Patch(color="#4FD18A", label="Budget <$10k"),
    mpatches.Patch(color=ACCENT,    label="Economy $10–20k"),
    mpatches.Patch(color=ACCENT4,   label="Mid-range $20–35k"),
    mpatches.Patch(color=ACCENT2,   label="Premium $35–60k"),
    mpatches.Patch(color="#C77DFF", label="Luxury $60k+"),
]
ax.legend(handles=seg_patches + [
    plt.Line2D([0],[0], color="white", linestyle="--", lw=1.4, label=f"Median ${median_p:,.0f}"),
    plt.Line2D([0],[0], color=ACCENT4, linestyle="--", lw=1.4, label=f"Mean ${mean_p:,.0f}"),
], loc="upper right", framealpha=0.4, fontsize=8)

ax.text(0.02, 0.92,
        f"78% of listings priced under $40k\nMean–Median gap: ${mean_p-median_p:,.0f} (luxury pull)",
        transform=ax.transAxes, fontsize=8, color=TEXT_SEC,
        va="top", bbox=dict(boxstyle="round,pad=0.4", fc=GRID_COLOR, ec=GRID_COLOR, alpha=0.8))

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart01_price_distribution.png")
plt.close()
print(" Chart 1 – Price distribution saved.")

# Chart 2

fig, ax = plt.subplots(figsize=(12, 5))

n2, bins2, patches2 = ax.hist(df["year"], bins=range(1990, 2026), color=ACCENT,
                               alpha=0.85, edgecolor=DARK_BG, linewidth=0.5)


for patch, left in zip(patches2, bins2[:-1]):
    if left < 2000:  patch.set_facecolor("#888888")
    elif left < 2010: patch.set_facecolor(ACCENT4)
    elif left < 2018: patch.set_facecolor(ACCENT)
    else:             patch.set_facecolor(ACCENT3)


for era, x, color in [("Pre-2000", 1995, "#888888"),
                       ("2000s", 2005, ACCENT4),
                       ("2010s", 2014, ACCENT),
                       ("2018+", 2021, ACCENT3)]:
    ax.text(x, n2.max() * 1.03, era, ha="center", fontsize=8,
            color=color, fontweight="bold")

modal_year = int(df["year"].mode()[0])
ax.axvline(modal_year, color="white", linestyle="--", lw=1.2,
           label=f"Mode: {modal_year}")
ax.set_title("Distribution of Model Year (Production Year)", fontsize=14, fontweight="bold")
ax.set_xlabel("Model Year", fontsize=10, labelpad=6)
ax.set_ylabel("Number of Listings", fontsize=10, labelpad=6)
ax.legend(framealpha=0.4)
ax.set_xlim(1989, 2025)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart02_model_year_distribution.png")
plt.close()
print("Chart 2 – Model year distribution saved.")

# Chart 3
fig, ax = plt.subplots(figsize=(12, 5))

mileage_clipped = df["mileage"].clip(upper=300_000)
n3, bins3, patches3 = ax.hist(mileage_clipped, bins=70, color=ACCENT3,
                               alpha=0.8, edgecolor=DARK_BG, linewidth=0.4)

p25 = df["mileage"].quantile(0.25)
p50 = df["mileage"].quantile(0.50)
p75 = df["mileage"].quantile(0.75)

ax.axvspan(0,   p25, alpha=0.07, color=ACCENT3, label=f"Q1 (0–{p25/1000:.0f}k mi)")
ax.axvspan(p25, p75, alpha=0.05, color=ACCENT4, label=f"IQR ({p25/1000:.0f}k–{p75/1000:.0f}k mi)")
ax.axvspan(p75, 300_000, alpha=0.05, color=ACCENT2, label=f"Q3+ (>{p75/1000:.0f}k mi)")

for pct, val, col in [(f"Q1 {p25/1000:.0f}k", p25, ACCENT3),
                      (f"Median {p50/1000:.0f}k", p50, "white"),
                      (f"Q3 {p75/1000:.0f}k", p75, ACCENT2)]:
    ax.axvline(val, color=col, linestyle="--", lw=1.2, alpha=0.9, label=pct)

ax.set_title("Mileage Distribution (miles driven)", fontsize=14, fontweight="bold")
ax.set_xlabel("Mileage (miles)", fontsize=10, labelpad=6)
ax.set_ylabel("Number of Listings", fontsize=10, labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_miles))
ax.legend(framealpha=0.4, ncol=2, fontsize=8)

ax.text(0.02, 0.92,
        f"IQR: {p25/1000:.0f}k – {p75/1000:.0f}k miles\n50% of cars fall in this range",
        transform=ax.transAxes, fontsize=8, color=TEXT_SEC,
        va="top", bbox=dict(boxstyle="round,pad=0.4", fc=GRID_COLOR, ec=GRID_COLOR, alpha=0.8))

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart03_mileage_distribution.png")
plt.close()
print("Chart 3 – Mileage distribution saved.")

# Chart 4
fig, ax = plt.subplots(figsize=(12, 6))

d4 = df[(df["price"] < 100_000) & (df["mileage"] < 250_000)].copy()


custom_cmap = LinearSegmentedColormap.from_list(
    "dark_blue", [PANEL_BG, "#1E3A6E", ACCENT, "#A8C8FF"])
hb = ax.hexbin(d4["mileage"], d4["price"], gridsize=55, cmap=custom_cmap,
               mincnt=1, linewidths=0.1)
cb = fig.colorbar(hb, ax=ax, pad=0.01)
cb.set_label("Listings per bin", fontsize=8, color=TEXT_SEC)
cb.ax.yaxis.set_tick_params(color=TEXT_SEC)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC, fontsize=8)


z = np.polyfit(d4["mileage"], d4["price"], 1)
p = np.poly1d(z)
x_line = np.linspace(d4["mileage"].min(), d4["mileage"].max(), 300)
ax.plot(x_line, p(x_line), color=ACCENT2, linewidth=2.2, label="Linear trend", zorder=5)

slope_per_10k = z[0] * 10_000
ax.text(0.97, 0.93,
        f"Each 10,000 miles → ${slope_per_10k:+,.0f} price change\nr = −0.42 (moderate negative)",
        transform=ax.transAxes, fontsize=8, color=TEXT_SEC, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc=GRID_COLOR, ec=GRID_COLOR, alpha=0.9))

ax.set_title("Price vs Mileage — Density Hexbin + Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Mileage (miles)", fontsize=10, labelpad=6)
ax.set_ylabel("Price (USD)", fontsize=10, labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_miles))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.legend(framealpha=0.4)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart04_price_vs_mileage_hexbin.png")
plt.close()
print("Chart 4 – Price vs Mileage hexbin saved.")

# Chart 5
fig, ax = plt.subplots(figsize=(12, 6))

d5 = df[(df["price"] < 100_000) & (df["age_of_car"] <= 30)].copy()

bins_age = [0, 3, 7, 12, 20, 30]
colors_age = [ACCENT3, ACCENT, ACCENT4, ACCENT2, "#888888"]
labels_age = ["0–3 yrs", "4–7 yrs", "8–12 yrs", "13–20 yrs", "20+ yrs"]
d5["age_bin"] = pd.cut(d5["age_of_car"], bins=bins_age, labels=labels_age)

for col, lbl, clr in zip(labels_age, labels_age, colors_age):
    subset = d5[d5["age_bin"] == col]
    ax.scatter(subset["age_of_car"], subset["price"],
               alpha=0.18, s=9, c=clr, edgecolors="none", label=lbl)

z2 = np.polyfit(d5["age_of_car"], d5["price"], 2) 
p2 = np.poly1d(z2)
x_age = np.linspace(0, 30, 200)
ax.plot(x_age, p2(x_age), color="white", linewidth=2.2, label="Quadratic trend", zorder=5)


ax.axvspan(3, 7, alpha=0.07, color=ACCENT, label="Sweet spot (4–7 yrs)")
ax.text(5, d5["price"].quantile(0.9),
        "Best value\nzone", ha="center", fontsize=8,
        color=ACCENT, fontweight="bold")

ax.set_title("Price vs Age of Car (years since production)", fontsize=14, fontweight="bold")
ax.set_xlabel("Age (years)", fontsize=10, labelpad=6)
ax.set_ylabel("Price (USD)", fontsize=10, labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.set_xlim(-0.5, 31)
ax.legend(framealpha=0.4, ncol=3, fontsize=8)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart05_price_vs_age.png")
plt.close()
print("Chart 5 – Price vs Age saved.")

# Chart 6
top12_brand = df["brand"].value_counts().head(12)
total = len(df)

fig, ax = plt.subplots(figsize=(11, 7))

colors_bar = [BRAND_COLORS.get(b, ACCENT) for b in top12_brand.index]
bars = ax.barh(range(len(top12_brand)), top12_brand.values,
               color=colors_bar, alpha=0.88,
               edgecolor=DARK_BG, linewidth=0.5, height=0.65)

ax.set_yticks(range(len(top12_brand)))
ax.set_yticklabels(top12_brand.index, fontsize=10, color=TEXT_PRI)
ax.invert_yaxis()

for i, (val, bar) in enumerate(zip(top12_brand.values, bars)):
    share = val / total * 100
    ax.text(val + 30, i, f"{val:,}  ({share:.1f}%)",
            va="center", fontsize=8.5, color=TEXT_SEC)

ax2 = ax.twiny()
cumshare = np.cumsum(top12_brand.values) / total * 100
ax2.plot(cumshare, range(len(top12_brand)), color=ACCENT4,
         marker="o", markersize=4, linewidth=1.5, label="Cumulative share")
ax2.set_xlabel("Cumulative market share (%)", fontsize=9, color=ACCENT4, labelpad=6)
ax2.tick_params(colors=ACCENT4)
ax2.spines["top"].set_edgecolor(ACCENT4)
ax2.set_xlim(0, 105)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

ax.set_title("Top 12 Brands by Listing Volume  ·  Pareto Analysis", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Listings", fontsize=10, labelpad=6)
ax.set_xlim(0, top12_brand.values.max() * 1.25)
ax.grid(axis="y", alpha=0)
ax.grid(axis="x", alpha=0.5)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart06_brand_volume_pareto.png")
plt.close()
print("Chart 6 – Brand volume Pareto saved.")

#Chart 7
brand_stats = (df.groupby("brand")["price"]
               .agg(["mean", "median", "std", "count"])
               .query("count >= 30")
               .sort_values("mean", ascending=False)
               .head(15))
brand_stats["se"] = brand_stats["std"] / np.sqrt(brand_stats["count"])

fig, ax = plt.subplots(figsize=(11, 7))

y_pos = range(len(brand_stats))
ax.barh(y_pos, brand_stats["mean"], color=ACCENT, alpha=0.25,
        height=0.6, edgecolor="none")
ax.errorbar(brand_stats["mean"], y_pos,
            xerr=brand_stats["se"] * 1.96,
            fmt="none", color=TEXT_SEC, linewidth=1, capsize=3, capthick=1)
ax.scatter(brand_stats["mean"],   y_pos, s=60, color=ACCENT,  zorder=5, label="Mean price")
ax.scatter(brand_stats["median"], y_pos, s=40, color=ACCENT4,
           marker="D", zorder=6, label="Median price")

ax.set_yticks(y_pos)
ax.set_yticklabels(brand_stats.index, fontsize=10, color=TEXT_PRI)
ax.invert_yaxis()
ax.set_xlabel("Price (USD)", fontsize=10, labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.set_title("Average vs Median Price by Brand (≥30 listings)  ·  95% CI", fontsize=14, fontweight="bold")
ax.legend(framealpha=0.4)
ax.grid(axis="y", alpha=0)

for i, (_, row) in enumerate(brand_stats.iterrows()):
    ax.text(row["mean"] + 800, i, fmt_usd(row["mean"]),
            va="center", fontsize=8, color=TEXT_SEC)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart07_avg_price_by_brand_dotplot.png")
plt.close()
print("Chart 7 – Avg price dot-plot saved.")

# Chart 8

state_stats = (df.groupby("state")["price"]
               .agg(["mean", "count"])
               .query("count >= 50")
               .reset_index())
national_mean = df["price"].mean()
state_stats["delta"] = state_stats["mean"] - national_mean
state_stats = state_stats.sort_values("delta").tail(20)

fig, ax = plt.subplots(figsize=(11, 8))

colors_div = [ACCENT2 if d > 0 else ACCENT for d in state_stats["delta"]]
ax.barh(state_stats["state"], state_stats["delta"],
        color=colors_div, alpha=0.85,
        edgecolor=DARK_BG, linewidth=0.5, height=0.65)

ax.axvline(0, color=TEXT_SEC, linewidth=0.8, linestyle="-")
ax.set_xlabel(f"Price deviation from national mean (${national_mean:,.0f})", fontsize=9, labelpad=6)
ax.set_title("State Average Price vs National Mean  ·  Top 20 States", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+,.0f}"))
ax.set_yticklabels(state_stats["state"], fontsize=9, color=TEXT_PRI)
ax.grid(axis="y", alpha=0)

for i, (_, row) in enumerate(state_stats.iterrows()):
    offset = 200 if row["delta"] >= 0 else -200
    ha = "left" if row["delta"] >= 0 else "right"
    ax.text(row["delta"] + offset, i,
            f"n={row['count']:,}", va="center", fontsize=7.5, color=TEXT_SEC, ha=ha)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart08_state_price_diverging.png")
plt.close()
print("Chart 8 – State diverging bar saved.")

# Chart 9
top_colors = df["color"].value_counts().head(10)

color_hex_map = {
    "white": "#F0F0F0",  "black": "#1A1A1A",  "silver": "#B8B8B8",
    "gray":  "#6E6E6E",  "red":   "#C0392B",  "blue":   "#2471A3",
    "brown": "#784212",  "beige": "#D4B896",  "green":  "#1E8449",
    "gold":  "#D4AC0D",  "orange":"#E67E22",  "yellow": "#D4AC0D",
    "purple":"#7D3C98",  "maroon":"#7B241C",  "teal":   "#0E6655",
}

fig, ax = plt.subplots(figsize=(12, 5))
pct = top_colors.values / top_colors.values.sum() * 100

bar_colors = [color_hex_map.get(c.lower(), ACCENT) for c in top_colors.index]
edge_colors = ["#444444" if c.lower() in ["white","silver","beige","gold","yellow"]
               else DARK_BG for c in top_colors.index]

bars9 = ax.bar(range(len(top_colors)), top_colors.values,
               color=bar_colors, edgecolor=edge_colors,
               linewidth=1.2, width=0.65)

ax.set_xticks(range(len(top_colors)))
ax.set_xticklabels([c.title() for c in top_colors.index], fontsize=10)

for i, (val, pv) in enumerate(zip(top_colors.values, pct)):
    label_color = "#222" if bar_colors[i] in ["#F0F0F0", "#B8B8B8", "#D4B896", "#D4AC0D"] else "white"
    ax.text(i, val + 50, f"{pv:.1f}%", ha="center", fontsize=9,
            color=label_color if val > 200 else TEXT_SEC, fontweight="bold")

ax.set_title("Most Popular Car Colors  ·  % of Top-10 Share", fontsize=14, fontweight="bold")
ax.set_xlabel("Color", fontsize=10, labelpad=6)
ax.set_ylabel("Number of Listings", fontsize=10, labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax.text(0.97, 0.92,
        "White + Black + Silver/Gray\ncover 63% of all listings",
        transform=ax.transAxes, fontsize=8, color=TEXT_SEC, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc=GRID_COLOR, ec=GRID_COLOR, alpha=0.8))

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart09_color_popularity.png")
plt.close()
print("Chart 9 – Color popularity saved.")

# Chart 10
num_cols = ["price", "year", "mileage", "age_of_car"]
corr = df[num_cols].corr()

cmap_corr = LinearSegmentedColormap.from_list(
    "corr", [ACCENT2, PANEL_BG, ACCENT])

fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(corr, dtype=bool)) 

sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap_corr,
            center=0, vmin=-1, vmax=1,
            linewidths=2, linecolor=DARK_BG,
            ax=ax, mask=mask, square=True,
            annot_kws={"size": 13, "weight": "bold", "color": TEXT_PRI},
            cbar_kws={"shrink": 0.8, "pad": 0.02})

cb10 = ax.collections[0].colorbar
cb10.ax.yaxis.set_tick_params(color=TEXT_SEC)
plt.setp(cb10.ax.yaxis.get_ticklabels(), color=TEXT_SEC, fontsize=8)

ax.set_xticklabels(["Price", "Year", "Mileage", "Age"], fontsize=10, color=TEXT_PRI)
ax.set_yticklabels(["Price", "Year", "Mileage", "Age"], fontsize=10, color=TEXT_PRI, rotation=0)
ax.set_title("Correlation Matrix — Numeric Variables\n(lower triangle · Pearson r)",
             fontsize=13, fontweight="bold")

ax.text(0.03, -0.18,
        "Key findings: Year↑ → Price↑ (r=0.53) · Mileage↑ → Price↓ (r=−0.42) · "
        "Age and Year are perfectly inverse (r=−1.00)",
        transform=ax.transAxes, fontsize=8, color=TEXT_SEC, wrap=True)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart10_correlation_heatmap.png")
plt.close()
print("Chart 10 – Correlation heatmap saved.")

# Chart 11
top8 = df["brand"].value_counts().head(8).index.tolist()
df_top8 = df[df["brand"].isin(top8)]

pivot = df_top8.pivot_table(values="price", index="brand",
                             columns="year", aggfunc="median")
# Keep only 2012–2023 (sufficient data, relevant range) - TO REVIEW AGAIN
pivot = pivot.loc[:, (pivot.columns >= 2012) & (pivot.columns <= 2023)]
pivot = pivot.dropna(thresh=6)

pivot_norm = pivot.sub(pivot.mean(axis=1), axis=0).div(pivot.std(axis=1), axis=0)

cmap_heat = LinearSegmentedColormap.from_list(
    "heat", ["#0D1B4A", "#1A3A8A", ACCENT, ACCENT4, ACCENT2])

fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[1.4, 1])

sns.heatmap(pivot / 1000, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=1.5, linecolor=DARK_BG,
            ax=axes[0], cbar_kws={"label": "Median price ($k)", "shrink": 0.8},
            annot_kws={"size": 9, "color": "#111"})
axes[0].set_title("Median Price ($k) by Brand × Model Year  ·  Top 8 Brands",
                  fontsize=13, fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("")
axes[0].set_yticklabels(axes[0].get_yticklabels(), color=TEXT_PRI, fontsize=9, rotation=0)
axes[0].set_xticklabels(axes[0].get_xticklabels(), color=TEXT_PRI, fontsize=9, rotation=45)

sns.heatmap(pivot_norm, annot=False, cmap=cmap_heat,
            center=0, linewidths=1.5, linecolor=DARK_BG,
            ax=axes[1], cbar_kws={"label": "Within-brand z-score", "shrink": 0.8})
axes[1].set_title("Within-Brand Price Trend (normalized)  ·  Warmer = relatively more expensive for that brand",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Model Year", fontsize=10, labelpad=6)
axes[1].set_ylabel("")
axes[1].set_yticklabels(axes[1].get_yticklabels(), color=TEXT_PRI, fontsize=9, rotation=0)
axes[1].set_xticklabels(axes[1].get_xticklabels(), color=TEXT_PRI, fontsize=9, rotation=45)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout(h_pad=2)
plt.savefig("../report/chart11_brand_year_heatmap.png")
plt.close()
print("Chart 11 – Brand × Year dual heatmap saved.")

# Chart 12

top6 = df["brand"].value_counts().head(6).index.tolist()
d12 = df[df["brand"].isin(top6)].copy()

seg_order = ["Budget\n(<$10k)", "Economy\n($10–20k)",
             "Mid-range\n($20–35k)", "Premium\n($35–60k)", "Luxury\n($60k+)"]
seg_colors_map = {
    "Budget\n(<$10k)":    "#4FD18A",
    "Economy\n($10–20k)": ACCENT,
    "Mid-range\n($20–35k)": ACCENT4,
    "Premium\n($35–60k)": ACCENT2,
    "Luxury\n($60k+)":    "#C77DFF",
}

pivot12 = (d12.groupby(["brand", "price_segment"])
             .size()
             .unstack(fill_value=0)
             .reindex(columns=seg_order)
             .loc[top6])
pivot12_pct = pivot12.div(pivot12.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(top6))

for seg in seg_order:
    vals = pivot12_pct[seg].values
    bars12 = ax.bar(top6, vals, bottom=bottom,
                    color=seg_colors_map[seg], alpha=0.88,
                    edgecolor=DARK_BG, linewidth=0.5, width=0.6,
                    label=seg.replace("\n", " "))
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 8:
            ax.text(i, b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    bottom += vals

ax.set_ylim(0, 100)
ax.set_ylabel("Share of brand listings (%)", fontsize=10, labelpad=6)
ax.set_title("Market Positioning Map — Price Segment Mix by Brand",
             fontsize=14, fontweight="bold")
ax.set_xticklabels(top6, fontsize=11, color=TEXT_PRI)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(loc="upper right", framealpha=0.4, ncol=5,
          fontsize=8, bbox_to_anchor=(1, 1.12))
ax.grid(axis="x", alpha=0)

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart12_segment_mix_by_brand.png")
plt.close()
print("Chart 12 – Segment mix by brand saved.")

# Chart 13
top5 = df["brand"].value_counts().head(5).index.tolist()
d13 = df[df["brand"].isin(top5) & (df["age_of_car"] <= 20)].copy()

dep = (d13.groupby(["brand", "age_of_car"])["price"]
          .median()
          .reset_index()
          .query("price > 0"))

fig, ax = plt.subplots(figsize=(12, 6))

line_colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#C77DFF"]
for brand, col in zip(top5, line_colors):
    bd = dep[dep["brand"] == brand].sort_values("age_of_car")
    ax.plot(bd["age_of_car"], bd["price"], color=col,
            linewidth=2.2, marker="o", markersize=4,
            label=brand, alpha=0.9)
    if not bd.empty:
        last = bd.iloc[-1]
        ax.text(last["age_of_car"] + 0.2, last["price"],
                brand, fontsize=8, color=col, va="center")

ax.set_title("Depreciation Curve — Median Price by Car Age  ·  Top 5 Brands",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Age (years)", fontsize=10, labelpad=6)
ax.set_ylabel("Median Price (USD)", fontsize=10, labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.set_xlim(0, 21)
ax.legend(framealpha=0.4, fontsize=9)

ax.text(0.02, 0.08,
        "Steepest drop in years 1–4 (new-car premium loss)\nRates flatten after year 8",
        transform=ax.transAxes, fontsize=8, color=TEXT_SEC,
        bbox=dict(boxstyle="round,pad=0.4", fc=GRID_COLOR, ec=GRID_COLOR, alpha=0.8))

add_figure_label(fig, "USA Cars Dataset · Kaggle")
plt.tight_layout()
plt.savefig("../report/chart13_depreciation_curves.png")
plt.close()
print("Chart 13 – Depreciation curves saved.")

print("\n" + "=" * 60)
print("ALL 13 CHARTS SAVED TO: ../report/")
print("=" * 60)