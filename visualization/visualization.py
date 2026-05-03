import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Design

DARK_BG    = "#0F1117"
PANEL_BG   = "#1A1D27"
GRID_COLOR = "#2A2D3A"
TEXT_PRI   = "#E8EAF0"
TEXT_SEC   = "#8A8FA8"
ACCENTS    = {
    "blue":   "#4F8EF7",
    "red":    "#F7674F",
    "green":  "#4FD18A",
    "yellow": "#F7C94F",
    "teal":   "#2EC4B6",
    "orange": "#FF9F1C",
    "purple": "#C77DFF",
    "crimson":"#F94144",
}
A = list(ACCENTS.values()) 

BRAND_COLORS = {
    "ford": A[0], "dodge": A[1], "nissan": A[2], "chevrolet": A[3],
    "gmc": A[4],  "jeep":  A[5], "chrysler": A[6], "bmw": A[7],
    "hyundai": "#90BE6D", "kia": "#F3722C", "buick": "#6FA8F9", "infiniti": "#F9D97A",
}

SEG_COLORS = {
    (0,      10_000): A[2],
    (10_000, 20_000): A[0],
    (20_000, 35_000): A[3],
    (35_000, 60_000): A[1],
    (60_000,100_000): A[6],
}
SEG_LABELS = {
    A[2]: "Budget  under $10k",
    A[0]: "Economy  $10-20k",
    A[3]: "Mid-range  $20-35k",
    A[1]: "Premium  $35-60k",
    A[6]: "Luxury  $60k+",
}
COLOR_HEX = {
    "white": "#F0F0F0", "black": "#1A1A1A", "silver": "#B8B8B8", "gray":    "#6E6E6E",
    "red":   "#C0392B", "blue":  "#2471A3", "brown":  "#784212", "beige":   "#D4B896",
    "green": "#1E8449", "gold":  "#D4AC0D", "orange": "#E67E22", "charcoal":"#4A4A4A",
}
LIGHT_COLORS = {"white", "silver", "beige", "gold"}

# Global style
plt.rcParams.update({
    # figure / axes backgrounds
    "figure.facecolor": DARK_BG,   "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":   GRID_COLOR,"axes.labelcolor":   TEXT_SEC,
    # titles
    "axes.titlecolor":  TEXT_PRI,  "axes.titlesize":    13,
    "axes.titleweight": "bold",    "axes.titlepad":     14,
    # grid
    "axes.grid":        True,      "grid.color":        GRID_COLOR,
    "grid.linewidth":   0.6,       "grid.alpha":        1.0,
    # spines
    "axes.spines.top":  False,     "axes.spines.right": False,
    "axes.spines.left": False,     "axes.spines.bottom":False,
    # ticks / legend
    "xtick.color":      TEXT_SEC,  "ytick.color":       TEXT_SEC,
    "xtick.labelsize":  9,         "ytick.labelsize":   9,
    "legend.facecolor": PANEL_BG,  "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor":TEXT_SEC,  "legend.fontsize":   9,
    # output
    "figure.dpi":       150,       "savefig.dpi":       150,
    "savefig.bbox":     "tight",   "savefig.facecolor": DARK_BG,
    "font.family":      "DejaVu Sans", "text.color":    TEXT_PRI,
})

# Helpers
def fmt_usd(x, _=None):
    if x >= 1_000_000: return f"${x/1e6:.1f}M"
    if x >= 1_000:     return f"${x/1000:.0f}k"
    return f"${x:.0f}"

def fmt_miles(x, _=None):
    return f"{x/1000:.0f}k mi" if x >= 1_000 else f"{x:.0f} mi"

def callout(ax, text, x=0.03, y=0.93):
    """Plain-English insight box."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, color=TEXT_PRI,
            va="top", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="#252838", ec=GRID_COLOR, alpha=0.95))

def save(fig, filename, note="USA Cars Dataset - Kaggle"):
    """Add footer label, tighten layout, save and close."""
    fig.text(0.98, 0.01, note, ha="right", va="bottom", fontsize=7, color=TEXT_SEC, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"report/{filename}")
    plt.close()
    print(f"Saved: {filename}")

def vline(ax, x, color, label):
    ax.axvline(x, color=color, linestyle="--", lw=1.6, label=label)

# ── Load & clean
df = pd.read_csv("../data/featured_cars.csv", encoding="latin1")
df.drop_duplicates(inplace=True)
df.dropna(subset=["price", "brand", "year", "mileage", "color", "state"], inplace=True)
df = df[(df["price"] > 0) & (df["mileage"] > 0)]

df_colors = df[df["color"] != "no_color"].copy()

df["price_segment"] = pd.cut(
    df["price"],
    bins=[0, 10_000, 20_000, 35_000, 60_000, 1_000_000],
    labels=["Budget (<$10k)", "Economy ($10-20k)", "Mid-range ($20-35k)",
            "Premium ($35-60k)", "Luxury ($60k+)"],
)
df["salvage_flag"] = df["title_status"].str.lower().str.contains("salvage").astype(int)

print(f"Dataset loaded: {df.shape[0]:,} records")

# Chart 1 — Price distribution 
fig, ax = plt.subplots(figsize=(12, 6))

n, bins, patches = ax.hist(
    df["price"].clip(upper=100_000), bins=80,
    color=A[0], alpha=0.85, edgecolor=DARK_BG, linewidth=0.4,
)
for patch, left in zip(patches, bins[:-1]):
    for (lo, hi), col in SEG_COLORS.items():
        if lo <= left < hi:
            patch.set_facecolor(col)
            patch.set_alpha(0.8)

median_p, mean_p = df["price"].median(), df["price"].mean()
vline(ax, median_p, "white", f"Typical price  ${median_p:,.0f}")
vline(ax, mean_p,   A[3],   f"Average price  ${mean_p:,.0f}")
ax.annotate(f"${median_p/1000:.1f}k\ntypical",
            xy=(median_p, n.max() * 0.78), xytext=(median_p + 1200, n.max() * 0.78),
            color="white", fontsize=9, fontweight="bold", ha="left")

seg_patches = [mpatches.Patch(color=col, label=lbl) for col, lbl in SEG_LABELS.items()]
ax.legend(
    handles=seg_patches + [
        plt.Line2D([0], [0], color="white", linestyle="--", lw=1.4, label=f"Typical  ${median_p:,.0f}"),
        plt.Line2D([0], [0], color=A[3],   linestyle="--", lw=1.4, label=f"Average  ${mean_p:,.0f}"),
    ],
    loc="upper right", framealpha=0.4, fontsize=8,
)
ax.set_title("What Do Most Used Cars Cost?", fontsize=15)
ax.set_xlabel("Listing Price", labelpad=6)
ax.set_ylabel("Number of Listings", labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
callout(ax, f"Most listings (94%) are priced under $40,000\n"
            f"The typical used car sells for ${median_p:,.0f}\n"
            f"The average is higher (${mean_p:,.0f}) because a few luxury cars pull the number up")
save(fig, "chart01_price_distribution.png")

# Chart 2 — Model year distribution 
fig, ax = plt.subplots(figsize=(12, 5))

n2, bins2, patches2 = ax.hist(
    df["year"], bins=range(1984, 2022),
    color=A[0], alpha=0.85, edgecolor=DARK_BG, linewidth=0.5,
)
era_cfg = [
    (lambda y: y < 2000,  "#888888", 1991, "Before 2000"),
    (lambda y: y < 2010,  A[3],      2005, "2000s"),
    (lambda y: y < 2018,  A[0],      2014, "2010s"),
    (lambda y: True,      A[2],      2019, "2018-2020"),
]
for patch, left in zip(patches2, bins2[:-1]):
    for condition, col, _, _ in era_cfg:
        if condition(left):
            patch.set_facecolor(col)
            break

for _, col, x, label in era_cfg:
    ax.text(x, n2.max() * 1.05, label, ha="center", fontsize=8, color=col, fontweight="bold")

modal_year = int(df["year"].mode()[0])
vline(ax, modal_year, "white", f"Most common year: {modal_year}")
ax.set_title("What Year Were These Cars Made?", fontsize=15)
ax.set_xlabel("Model Year", labelpad=6)
ax.set_ylabel("Number of Listings", labelpad=6)
ax.set_xlim(1983, 2021)
ax.legend(framealpha=0.4, fontsize=9)
callout(ax, f"The most common model year is {modal_year}\n"
            f"Most listings are 2015 or newer\n"
            f"Very few cars older than 2005 are being sold", y=0.88)
save(fig, "chart02_model_year_distribution.png")

# Chart 3 — Mileage distribution
fig, ax = plt.subplots(figsize=(12, 5))

ax.hist(df["mileage"].clip(upper=300_000), bins=70,
        color=A[2], alpha=0.8, edgecolor=DARK_BG, linewidth=0.4)

p25, p50, p75 = df["mileage"].quantile([0.25, 0.50, 0.75])
ax.axvspan(0,      p25,     alpha=0.10, color=A[2])
ax.axvspan(p25,    p75,     alpha=0.07, color=A[3])
ax.axvspan(p75, 300_000,    alpha=0.07, color=A[1])
vline(ax, p50, "white", f"Typical: {p50/1000:.0f},000 miles")

ax.set_title("How Many Miles Have These Cars Been Driven?", fontsize=15)
ax.set_xlabel("Miles Driven", labelpad=6)
ax.set_ylabel("Number of Listings", labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_miles))
ax.legend(framealpha=0.4, fontsize=9)
callout(ax, f"Typical car has been driven {p50/1000:.0f},000 miles\n"
            f"Half of all listings are between {p25/1000:.0f},000-{p75/1000:.0f},000 miles\n"
            f"Fewer than 25% have driven over {p75/1000:.0f},000 miles")
save(fig, "chart03_mileage_distribution.png")

# Chart 4 — Price vs mileage hexbin
fig, ax = plt.subplots(figsize=(12, 6))

d4 = df[(df["price"] < 100_000) & (df["mileage"] < 250_000)]
cmap = LinearSegmentedColormap.from_list("dark_blue", [PANEL_BG, "#1E3A6E", A[0], "#A8C8FF"])
hb = ax.hexbin(d4["mileage"], d4["price"], gridsize=50, cmap=cmap, mincnt=1, linewidths=0.1)

cb = fig.colorbar(hb, ax=ax, pad=0.01)
cb.set_label("Number of listings in this zone", fontsize=8, color=TEXT_SEC)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_SEC, fontsize=8)

z = np.polyfit(d4["mileage"], d4["price"], 1)
x_line = np.linspace(d4["mileage"].min(), d4["mileage"].max(), 300)
ax.plot(x_line, np.poly1d(z)(x_line), color=A[1], linewidth=2.5, label="Price trend line", zorder=5)

slope_per_10k = z[0] * 10_000
ax.set_title("Do Cars With More Miles Cost Less?  (Yes)", fontsize=15)
ax.set_xlabel("Miles Driven", labelpad=6)
ax.set_ylabel("Listing Price", labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_miles(x)))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.legend(framealpha=0.4, fontsize=9)
callout(ax, f"Every 10,000 extra miles = ${abs(slope_per_10k):,.0f} lower price on average\n"
            f"Brighter areas show where most listings cluster\n"
            f"Low-mileage cars vary widely — brand matters too")
save(fig, "chart04_price_vs_mileage_hexbin.png")

# Chart 5 — Price vs age scatter 
fig, ax = plt.subplots(figsize=(12, 6))

d5 = df[(df["price"] < 100_000) & (df["car_age"] <= 20)].copy()
age_bins = pd.cut(d5["car_age"], bins=[5, 8, 12, 16, 20],
                  labels=["6-8 yrs", "9-12 yrs", "13-16 yrs", "17-20 yrs"])
for label, col in zip(age_bins.cat.categories, A[:4]):
    s = d5[age_bins == label]
    ax.scatter(s["car_age"], s["price"], alpha=0.15, s=9, c=col, edgecolors="none", label=label)

x_age = np.linspace(d5["car_age"].min(), d5["car_age"].max(), 200)
ax.plot(x_age, np.poly1d(np.polyfit(d5["car_age"], d5["price"], 2))(x_age),
        color="white", linewidth=2.5, label="Average price curve", zorder=5)

ax.set_title("Do Older Cars Cost Less?  (Yes, Consistently)", fontsize=15)
ax.set_xlabel("Age of Car (years old)", labelpad=6)
ax.set_ylabel("Listing Price", labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.set_xlim(5, 21)
ax.legend(framealpha=0.4, ncol=2, fontsize=8)
callout(ax, "Price drops steadily as a car gets older\n"
            "The sharpest drop happens between ages 9 and 14\n"
            "After age 14 prices are low but level off")
save(fig, "chart05_price_vs_age.png")

# Chart 6 — Brand volume 
top12_brand = df["brand"].value_counts().head(12)
total = len(df)

fig, ax = plt.subplots(figsize=(11, 7))
colors_bar = [BRAND_COLORS.get(b, A[0]) for b in top12_brand.index]
ax.barh(range(len(top12_brand)), top12_brand.values,
        color=colors_bar, alpha=0.88, edgecolor=DARK_BG, linewidth=0.5, height=0.65)
ax.set_yticks(range(len(top12_brand)))
ax.set_yticklabels([b.title() for b in top12_brand.index], fontsize=10, color=TEXT_PRI)
ax.invert_yaxis()

for i, val in enumerate(top12_brand.values):
    ax.text(val + 10, i, f"{val:,}  ({val/total*100:.0f}%)", va="center", fontsize=8.5, color=TEXT_SEC)

ford_pct  = top12_brand.iloc[0] / total * 100
top4_pct  = top12_brand.head(4).sum() / total * 100
ax.set_title("Which Car Brands Have the Most Listings?", fontsize=15)
ax.set_xlabel("Number of Listings", labelpad=6)
ax.set_xlim(0, top12_brand.values.max() * 1.35)
ax.grid(axis="y", alpha=0)
callout(ax, f"Ford dominates with {ford_pct:.0f}% of all listings\n"
            f"Ford, Dodge, Nissan & Chevrolet together make up\n"
            f"{top4_pct:.0f}% of the entire used car market here", x=0.45, y=0.25)
save(fig, "chart06_brand_volume_pareto.png")

# Chart 7 — Average price by brand
brand_stats = (
    df.groupby("brand")["price"]
    .agg(["mean", "median", "std", "count"])
    .query("count >= 10")
    .sort_values("median", ascending=False)
)

fig, ax = plt.subplots(figsize=(11, 7))
y_pos = range(len(brand_stats))
ax.barh(y_pos, brand_stats["median"], color=A[0], alpha=0.30, height=0.65, edgecolor="none")
ax.scatter(brand_stats["median"], y_pos, s=80, color=A[0], zorder=5, label="Typical price (median)")
ax.set_yticks(y_pos)
ax.set_yticklabels([b.title() for b in brand_stats.index], fontsize=10, color=TEXT_PRI)
ax.invert_yaxis()

for i, (_, row) in enumerate(brand_stats.iterrows()):
    ax.text(row["median"] + 200, i, fmt_usd(row["median"]), va="center", fontsize=8.5, color=TEXT_SEC)

top_brand, bot_brand = brand_stats.index[0].title(), brand_stats.index[-1].title()
top_price, bot_price = brand_stats["median"].iloc[0], brand_stats["median"].iloc[-1]
ax.set_title("Which Brands Are Most Expensive?", fontsize=15)
ax.set_xlabel("Typical Listing Price", labelpad=6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.grid(axis="y", alpha=0)
ax.legend(framealpha=0.4, fontsize=9)
callout(ax, f"{top_brand} listings tend to be the priciest at a typical {fmt_usd(top_price)}\n"
            f"{bot_brand} listings are the most affordable at around {fmt_usd(bot_price)}", x=0.45, y=0.25)
save(fig, "chart07_avg_price_by_brand_dotplot.png")

# Chart 8 — State price diverging bar 
state_stats = (
    df.groupby("state")["price"]
    .agg(["mean", "count"])
    .query("count >= 30")
    .reset_index()
)
national_mean = df["price"].mean()
state_stats["delta"] = state_stats["mean"] - national_mean
state_stats = state_stats.sort_values("delta")

fig, ax = plt.subplots(figsize=(11, 8))
colors_div = [A[1] if d > 0 else A[0] for d in state_stats["delta"]]
ax.barh(state_stats["state"], state_stats["delta"],
        color=colors_div, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5, height=0.65)
ax.axvline(0, color=TEXT_SEC, linewidth=1.2)
ax.set_xlabel(f"Price vs national average  (${national_mean:,.0f})", fontsize=9, labelpad=6)
ax.set_title("Where Are Used Cars Cheaper or More Expensive?", fontsize=15)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+,.0f}"))
ax.set_yticklabels([s.title() for s in state_stats["state"]], fontsize=9, color=TEXT_PRI)
ax.grid(axis="y", alpha=0)

n_states = len(state_stats)
ax.text(state_stats["delta"].min() * 0.7, -0.8,       "CHEAPER", color=A[0], fontsize=8, fontweight="bold")
ax.text(state_stats["delta"].max() * 0.5, n_states - 0.2, "PRICIER", color=A[1], fontsize=8, fontweight="bold")

for i, (_, row) in enumerate(state_stats.iterrows()):
    offset, ha = (120, "left") if row["delta"] >= 0 else (-120, "right")
    ax.text(row["delta"] + offset, i, f"n={row['count']:,}", va="center", fontsize=7.5, color=TEXT_SEC, ha=ha)

cheapest = state_stats.iloc[0]["state"].title()
priciest = state_stats.iloc[-1]["state"].title()
callout(ax, f"Blue = below national average (cheaper)\n"
            f"Red = above national average (pricier)\n"
            f"{cheapest} has the lowest average prices\n"
            f"{priciest} has the highest average prices", x=0.50, y=0.25)
save(fig, "chart08_state_price_diverging.png")

# Chart 9 — Color popularity (pie)
DARK_SLICES = {"black", "red", "blue", "brown", "green", "charcoal"}

fig, ax = plt.subplots(figsize=(9, 7))

top_colors = df_colors["color"].value_counts().head(10)
bar_colors  = [COLOR_HEX.get(c.lower(), A[0]) for c in top_colors.index]
edge_colors = ["#555" if c.lower() in LIGHT_COLORS else DARK_BG for c in top_colors.index]

wedges, texts, autotexts = ax.pie(
    top_colors.values,
    labels=[c.title() for c in top_colors.index],
    colors=bar_colors,
    autopct="%1.0f%%",
    startangle=140,
    wedgeprops={"edgecolor": PANEL_BG, "linewidth": 1.5},
    pctdistance=0.80,
)

for t in texts:
    t.set_color(TEXT_PRI)
    t.set_fontsize(9)

for at, color_name in zip(autotexts, top_colors.index):
    at.set_color("white" if color_name.lower() in DARK_SLICES else "#222222")
    at.set_fontsize(8)
    at.set_fontweight("bold")

ax.set_title("What Colors Are Most Common?", fontsize=15)
callout(ax, f"78% e makinave janë Bardha, Zeza, Argjend ose Gri\n"
            f"E bardha është ngjyra më e popullarizuar\n"
            f"Ngjyrat e tjera zënë vetëm 22% të tregut",
        x=0.68, y=0.15)
save(fig, "chart09_color_popularity.png")

# Chart 10 — What affects price most
corr = df[["price", "year", "mileage", "car_age"]].corr()["price"]

factors    = ["Model Year\n(newer = pricier)", "Miles Driven\n(more miles = cheaper)", "Age of Car\n(older = cheaper)"]
raw_rs     = [corr["year"], corr["mileage"], corr["car_age"]]
strengths  = [abs(r) for r in raw_rs]
bar_colors = [A[2] if r > 0 else A[1] for r in raw_rs]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(factors, strengths, color=bar_colors, alpha=0.82, edgecolor=DARK_BG, height=0.5)
ax.set_xlim(0, 0.65)
ax.set_xlabel("Strength of influence on price  (0 = none, 1 = perfect)", fontsize=9, labelpad=6)
ax.set_title("What Affects a Used Car's Price the Most?", fontsize=15)
ax.grid(axis="y", alpha=0)

for bar, r in zip(bars, raw_rs):
    direction = "pushes price UP" if r > 0 else "pushes price DOWN"
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{abs(r):.2f}  ({direction})", va="center", fontsize=9, color=TEXT_SEC)

callout(ax, "All three factors have a similar influence on price\n"
            "Miles driven has the strongest single effect\n"
            "Green = more of it means higher price\n"
            "Red = more of it means lower price", x=0.55, y=0.40)
save(fig, "chart10_correlation_heatmap.png")

# Chart 11 — Brand × model year heatmap 
top8 = df["brand"].value_counts().head(8).index.tolist()
pivot = (
    df[df["brand"].isin(top8)]
    .pivot_table(values="price", index="brand", columns="year", aggfunc="median")
    .loc[:, lambda d: (d.columns >= 2012) & (d.columns <= 2020)]
    .dropna(thresh=4)
)
pivot.index = [b.title() for b in pivot.index]

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(pivot / 1000, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=1.5, linecolor=DARK_BG, ax=ax,
            cbar_kws={"label": "Typical price ($k)", "shrink": 0.7},
            annot_kws={"size": 10, "color": "#111"})
ax.set_title("Typical Price by Brand & Model Year  (in $thousands)", fontsize=15)
ax.set_xlabel("Model Year the car was made", labelpad=8)
ax.set_ylabel("")
ax.set_yticklabels(ax.get_yticklabels(), color=TEXT_PRI, fontsize=10, rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), color=TEXT_PRI, fontsize=9, rotation=45)
fig.text(0.13, -0.04,
         "Read: find a brand, scan across to any year for its typical price.  "
         "Darker = more expensive.  Empty = no data.",
         fontsize=8.5, color=TEXT_SEC)
save(fig, "chart11_brand_year_heatmap.png")

# Chart 12 — Segment mix by brand
SEG_ORDER = ["Budget (<$10k)", "Economy ($10-20k)", "Mid-range ($20-35k)", "Premium ($35-60k)", "Luxury ($60k+)"]
SEG_PALETTE = [A[2], A[0], A[3], A[1], A[6]]

top6 = df["brand"].value_counts().head(6).index.tolist()
d12  = df[df["brand"].isin(top6)].copy()
d12["price_segment"] = pd.cut(d12["price"],
    bins=[0, 10_000, 20_000, 35_000, 60_000, 1_000_000], labels=SEG_ORDER)

pivot12_pct = (
    d12.groupby(["brand", "price_segment"]).size()
    .unstack(fill_value=0)
    .reindex(columns=SEG_ORDER)
    .loc[top6]
    .pipe(lambda df: df.div(df.sum(axis=1), axis=0) * 100)
)

fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(top6))
x_pos  = np.arange(len(top6))

for seg, col in zip(SEG_ORDER, SEG_PALETTE):
    vals = pivot12_pct[seg].values
    ax.bar(x_pos, vals, bottom=bottom, color=col, alpha=0.88,
           edgecolor=DARK_BG, linewidth=0.5, width=0.6, label=seg)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 9:
            ax.text(x_pos[i], b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    bottom += vals

ax.set_xticks(x_pos)
ax.set_xticklabels([b.title() for b in top6], fontsize=11, color=TEXT_PRI)
ax.set_ylim(0, 100)
ax.set_ylabel("Percentage of brand's listings", labelpad=6)
ax.set_title("What Price Range Does Each Brand Mostly Sell In?", fontsize=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(loc="upper right", framealpha=0.4, ncol=5, fontsize=8, bbox_to_anchor=(1, 1.13))
ax.grid(axis="x", alpha=0)
callout(ax, "Each bar shows 100% of that brand's listings split by price\n"
            "Ford has the most premium & luxury listings\n"
            "Nissan & Jeep are mostly budget & economy")
save(fig, "chart12_segment_mix_by_brand.png")

# Chart 13 — Depreciation curves
top5 = df["brand"].value_counts().head(5).index.tolist()
dep = (
    df[df["brand"].isin(top5) & (df["car_age"] <= 20)]
    .groupby(["brand", "car_age"])["price"]
    .median()
    .reset_index()
    .query("price > 0")
)

fig, ax = plt.subplots(figsize=(12, 6))
for brand, col in zip(top5, A[:5]):
    bd = dep[dep["brand"] == brand].sort_values("car_age")
    ax.plot(bd["car_age"], bd["price"], color=col, linewidth=2.2,
            marker="o", markersize=4, label=brand.title(), alpha=0.9)
    if not bd.empty:
        last = bd.iloc[-1]
        ax.text(last["car_age"] + 0.2, last["price"], brand.title(), fontsize=8, color=col, va="center")

ax.set_title("How Fast Do Cars Lose Their Value?", fontsize=15)
ax.set_xlabel("How old the car is (years)", labelpad=6)
ax.set_ylabel("Typical Listing Price", labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.set_xlim(5, 21)
ax.legend(framealpha=0.4, fontsize=9)
callout(ax, "All brands lose value as they age — this is called depreciation\n"
            "The drop is steepest between ages 9 and 14\n"
            "Ford & Chevrolet hold their value a bit better than Nissan")
save(fig, "chart13_depreciation_curves.png")

# Chart 14 — Price per mile by mileage category
CAT_ORDER  = ["Low", "Medium", "High", "Very High"]
CAT_LABELS = ["Low mileage\n(under ~21k mi)", "Medium mileage\n(21k-60k mi)",
              "High mileage\n(60k-100k mi)", "Very High mileage\n(over 100k mi)"]

d14 = df[df["mileage_category"].notna() & (df["price_per_mile"] <= df["price_per_mile"].quantile(0.97))]
medians14 = [d14[d14["mileage_category"] == c]["price_per_mile"].median() for c in CAT_ORDER]
counts14  = [d14[d14["mileage_category"] == c].shape[0]               for c in CAT_ORDER]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(range(len(CAT_ORDER)), medians14, color=A[:4], alpha=0.85, edgecolor=DARK_BG, width=0.55)
ax.set_xticks(range(len(CAT_ORDER)))
ax.set_xticklabels(CAT_LABELS, fontsize=10, color=TEXT_PRI)
ax.set_ylabel("Typical price per mile driven  (USD)", labelpad=6)
ax.set_title("Low-Mileage Cars Cost Much More Per Mile Driven", fontsize=15)
ax.grid(axis="x", alpha=0)

for bar, med, n in zip(bars, medians14, counts14):
    ax.text(bar.get_x() + bar.get_width() / 2, med + 0.02,
            f"${med:.2f}/mile\n({n:,} listings)",
            ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

low_ppm, high_ppm = medians14[0], medians14[-1]
callout(ax, f"A low-mileage car costs ${low_ppm:.2f} for every mile it has driven\n"
            f"A very high-mileage car costs just ${high_ppm:.2f} per mile driven\n"
            f"This reflects the premium buyers pay for a 'fresh' car\n"
            f"More miles = lower price, but also lower cost-per-mile", x=0.55, y=0.93)
save(fig, "chart14_price_per_mile_by_category.png")

# Chart 15 — Price by age group, top 4 brands 
AGE_KEYS   = ["Recent", "Old", "Classic"]
AGE_LABELS = ["Recent\n(6-7 yrs old)", "Older\n(8-15 yrs old)", "Classic\n(16+ yrs old)"]

top4 = df["brand"].value_counts().head(4).index.tolist()
d15  = df[df["brand"].isin(top4)].copy()
d15["age_group"] = pd.Categorical(d15["age_group"], categories=AGE_KEYS, ordered=True)

pivot15 = (
    d15.groupby(["age_group", "brand"])["price"]
    .median()
    .unstack()
    .reindex(index=AGE_KEYS)
    .loc[:, top4]
)

x       = np.arange(len(AGE_KEYS))
width   = 0.18
offsets = np.linspace(-(len(top4) - 1) / 2 * width, (len(top4) - 1) / 2 * width, len(top4))

fig, ax = plt.subplots(figsize=(12, 6))
for brand, col, offset in zip(top4, A[:4], offsets):
    vals = pivot15[brand].values
    bars = ax.bar(x + offset, vals, width=width, color=col, alpha=0.85,
                  edgecolor=DARK_BG, linewidth=0.5, label=brand.title())
    for xi, v in zip(x + offset, vals):
        if not np.isnan(v):
            ax.text(xi, v + 250, fmt_usd(v), ha="center", fontsize=8, color=TEXT_SEC, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(AGE_LABELS, fontsize=11, color=TEXT_PRI)
ax.set_ylabel("Typical Listing Price", labelpad=6)
ax.set_title("How Age Affects Price — Top 4 Brands Compared", fontsize=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.legend(framealpha=0.4, fontsize=10, title="Brand")
callout(ax, "Recent cars (6-7 yrs old) are always the most expensive\n"
            "Older cars (8-15 yrs) lose a big chunk of their value\n"
            "Classic cars (16+ yrs) are the cheapest across all brands\n"
            "Ford & Chevrolet tend to hold value better than Nissan")
save(fig, "chart15_price_by_age_group_brand.png")

# Chart 16 — Brand tier analysis
TIER_ORDER  = ["Economy", "Mid", "Luxury"]
TIER_LABELS = [
    "Economy Brands\n(e.g. Acura, Mercedes*)",
    "Mid-Range Brands\n(e.g. Ford, Nissan, Dodge)",
    "Luxury Brands\n(e.g. BMW, Audi, Lexus)",
]
d16 = df[df["price"] < 90_000]
medians16 = [d16[d16["brand_tier"] == t]["price"].median() for t in TIER_ORDER]
counts16  = [d16[d16["brand_tier"] == t].shape[0]          for t in TIER_ORDER]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.bar(range(len(TIER_ORDER)), medians16, color=A[:3], alpha=0.85, edgecolor=DARK_BG, width=0.5)
ax.set_xticks(range(len(TIER_ORDER)))
ax.set_xticklabels(TIER_LABELS, fontsize=10, color=TEXT_PRI)
ax.set_ylabel("Typical Listing Price", labelpad=6)
ax.set_title("Do Luxury Brands Actually Cost More Here?", fontsize=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
ax.grid(axis="x", alpha=0)

for bar, med, n in zip(bars, medians16, counts16):
    ax.text(bar.get_x() + bar.get_width() / 2, med + 200,
            f"{fmt_usd(med)}\n({n:,} listings)",
            ha="center", va="bottom", fontsize=9.5, color="white", fontweight="bold")

callout(ax, "Surprisingly, the three tiers have similar median prices\n"
            "This is partly because the dataset has very few luxury listings\n"
            "Note: the 'Economy' tier grouping has a data error —\n"
            "it includes Mercedes-Benz & Acura (should be Luxury)", x=0.38, y=0.93)
save(fig, "chart16_brand_tier_analysis.png",
     note="USA Cars Dataset - Kaggle  *Economy tier label error in source data")

# Chart 17 — Salvage vs clean title
clean   = df[df["salvage_flag"] == 0]
salvage = df[df["salvage_flag"] == 1]

clean_vals = [clean["price"].median(),   clean["mileage"].median()]
sal_vals   = [salvage["price"].median(), salvage["mileage"].median()]
price_disc = (1 - sal_vals[0] / clean_vals[0]) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax_i, (metric, cv, sv, fmtr, ylabel) in enumerate(zip(
    ["Typical Price", "Typical Mileage"],
    clean_vals, sal_vals,
    [fmt_usd, lambda x: f"{x/1000:.0f}k miles"],
    ["Typical Listing Price", "Miles Driven"],
)):
    ax = axes[ax_i]
    b1 = ax.bar(0, cv, color=A[0], alpha=0.85, edgecolor=DARK_BG, width=0.5,
                label=f"Clean title  (n={len(clean):,})")
    b2 = ax.bar(1, sv, color=A[1], alpha=0.85, edgecolor=DARK_BG, width=0.5,
                label=f"Salvage title  (n={len(salvage):,})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean Title", "Salvage Title"], fontsize=11, color=TEXT_PRI)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="x", alpha=0)
    for bar, val in [(b1, cv), (b2, sv)]:
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, val * 1.02, fmtr(val),
                ha="center", va="bottom", fontsize=11, color="white", fontweight="bold")
    if ax_i == 0:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_usd))
        ax.legend(framealpha=0.4, fontsize=9, loc="upper right")
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

fig.suptitle("Salvage Title Cars: Much Cheaper, But Much Higher Mileage",
             fontsize=15, fontweight="bold", color=TEXT_PRI, y=1.01)
callout(axes[0],
        f"A salvage title means the car was declared a total loss\n"
        f"by an insurance company (accident, flood, etc.)\n"
        f"Salvage cars are {price_disc:.0f}% cheaper on average\n"
        f"but they've driven 3-4x more miles", y=0.60)
save(fig, "chart17_salvage_vs_clean.png",
     note="USA Cars Dataset - Kaggle  (salvage re-derived from title_status)")

# Chart 18 — Mileage category × age group 
CAT_ORDER2 = ["Low", "Medium", "High", "Very High"]
AGE_KEYS2  = ["Recent", "Old", "Classic"]
AGE_LABELS2 = ["Recent\n(6-7 yrs)", "Older\n(8-15 yrs)", "Classic\n(16+ yrs)"]

cross_pct = (
    df.groupby(["age_group", "mileage_category"]).size()
    .unstack(fill_value=0)
    .reindex(index=AGE_KEYS2, columns=CAT_ORDER2)
    .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100)
)

fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(AGE_KEYS2))
x_pos  = np.arange(len(AGE_KEYS2))

for cat, col in zip(CAT_ORDER2, A[:4]):
    vals = cross_pct[cat].values
    ax.bar(x_pos, vals, bottom=bottom, color=col, alpha=0.85,
           edgecolor=DARK_BG, width=0.55, label=cat + " mileage")
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 7:
            ax.text(x_pos[i], b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    bottom += vals

ax.set_xticks(x_pos)
ax.set_xticklabels(AGE_LABELS2, fontsize=11, color=TEXT_PRI)
ax.set_ylim(0, 100)
ax.set_ylabel("Percentage of listings", labelpad=6)
ax.set_title("Do Newer Cars Have Fewer Miles?  (Yes — Strongly)", fontsize=15)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.legend(loc="upper right", framealpha=0.4, fontsize=9)
ax.grid(axis="x", alpha=0)
callout(ax, "70% of recent cars (6-7 yrs old) have LOW mileage\n"
            "Older cars (8-15 yrs) are mostly medium mileage\n"
            "Classic cars (16+ yrs) are almost all high or very high mileage\n"
            "Age and mileage go hand in hand — older = more miles driven")
save(fig, "chart18_mileage_category_x_age_group.png")

print("\n" + "=" * 60)
print("ALL 18 CHARTS SAVED TO: report/")
print("=" * 60)