import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression

# Read in bikes.csv into a pandas dataframe
bikes = pd.read_csv("bikes.csv", index_col="bike_id")

# Read in DOX.csv into a pandas dataframe
# Be sure to parse the 'Date' column as a datetime
dox = pd.read_csv("DOX.csv", parse_dates=["Date"])

# Divide the figure into six subplots
# Divide the figure into subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Make a pie chart
bikes_status = (
    bikes["status"]
    .value_counts()
    .rename_axis("unique_values")
    .reset_index(name="counts")
)
patches_pie, text_pie, pcts_pie = axs[0][0].pie(
    bikes_status["counts"], labels=bikes_status["unique_values"], autopct="%1.1f%%"
)
for i, patch in enumerate(patches_pie):
    text_pie[i].set_color(patch.get_facecolor())
plt.setp(pcts_pie, color="white")
axs[0][0].set_title("Current Status")

# Make a histogram with quartile lines
# There should be 20 bins

axs[0][1].hist(bikes["purchase_price"], bins=20, histtype="step")
axs[0][1].set(
    xlabel="US Dollars", ylabel="Number of Bikes", title="Price Histogram (1000 bikes)"
)
quants = [
    bikes["purchase_price"].quantile(0.0),
    bikes["purchase_price"].quantile(0.25),
    bikes["purchase_price"].quantile(0.50),
    bikes["purchase_price"].quantile(0.75),
    bikes["purchase_price"].quantile(1.0),
]
for i in quants:
    axs[0][1].axvline(i, linestyle="--", color="k")

# Let's write some annotations
axs[0][1].text(quants[0] + 8, 10, f"min: ${quants[0]:.0f}", size=12, rotation=90)
axs[0][1].text(quants[1] + 8, 10, f"25%: ${quants[1]:.0f}", size=12, rotation=90)
axs[0][1].text(quants[2] + 8, 10, f"50%: ${quants[2]:.0f}", size=12, rotation=90)
axs[0][1].text(quants[3] + 8, 10, f"75%: ${quants[3]:.0f}", size=12, rotation=90)
axs[0][1].text(quants[4] + 8, 10, f"max: ${quants[4]:.0f}", size=12, rotation=90)
# Add the dollar sign
axs[0][1].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%d"))


# Make a scatter plot with a trend line
# Get data as numpy arrays
X = bikes["purchase_price"].values.reshape(-1, 1)
y = bikes["weight"].values.reshape(-1, 1)

axs[1][0].scatter(bikes["purchase_price"], bikes["weight"], s=3, alpha=0.4)
# Do linear regression
reg = LinearRegression()
reg.fit(X, y)
axs[1][0].plot(X, reg.predict(X), color="r")
axs[1][0].set(xlabel="Price", ylabel="Weight", title="Price vs. Weight")
axs[1][0].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%d"))
axs[1][0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d kg"))

# Make plot for DOX
axs[1][1].plot(dox["Date"], dox["Adj Close"])
axs[1][1].grid()
axs[1][1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%1.2f"))
axs[1][1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
axs[1][1].set_title("DOX")

# Make a boxplot sorted so mean values are increasing
# Hide outliers
grouped_bikes_df = pd.DataFrame(
    {col: val["purchase_price"] for col, val in bikes.groupby("brand")}
)
mean_sorted_index = grouped_bikes_df.mean().sort_values().index.to_list()
grouped_bikes_df[mean_sorted_index].boxplot(ax=axs[2][0], showfliers=False, grid=True)
axs[2][0].set_title("Brand vs. Price")
axs[2][0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%d"))


# Make a violin plot
axs[2][1].violinplot(
    dataset=[
        bikes.groupby(["brand"])["purchase_price"].apply(list)[idx]
        for idx in mean_sorted_index
    ],
    showmeans=True,
)
axs[2][1].set_title("Brand vs. Price")
axs[2][1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%d"))
axs[2][1].set_xticklabels([""] + mean_sorted_index)

# Create some space between subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Write out the plots as an image
plt.savefig("plots_ravi.png")
