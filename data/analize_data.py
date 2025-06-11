"""Analyze PC build price and performance data."""
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filepath):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(filepath)


def describe_prices(df):
    """Print descriptive statistics of the Total Price column."""
    print("Descriptive statistics for Total Price:")
    print(df["Total Price"].describe())


def plot_price_distribution(df):
    """Plot a histogram of Total Price."""
    plt.hist(df["Total Price"], bins=30, color="skyblue")
    plt.xlabel("Total Price ($)")
    plt.ylabel("Count")
    plt.title("Distribution of Build Prices")
    plt.tight_layout()
    plt.savefig("price_distribution.png")
    plt.close()


def count_builds_by_price(df):
    """Print the number of builds in predefined price ranges."""
    bins = [0, 1000, 1500, 2500, 4000]
    grouped = pd.cut(df["Total Price"], bins=bins)
    print("Number of builds by price range:")
    print(df.groupby(grouped).size())


def check_missing_values(df):
    """Print missing value counts and rows with missing video cards."""
    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nRows with missing video card values:")
    print(df[df["Video Card"].isnull()])


def show_correlation(df):
    """Print correlation matrix for price and scores."""
    print("\nCorrelation matrix:")
    print(df[["Total Price", "Game Score", "Work Score"]].corr())


def plot_price_score_relation(df):
    """Plot scatter plots of Total Price vs Game and Work Scores."""
    plt.scatter(df["Total Price"], df["Game Score"], alpha=0.5, label="Game Score")
    plt.scatter(df["Total Price"], df["Work Score"], alpha=0.5,
                label="Work Score", color="orange")
    plt.xlabel("Total Price ($)")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Price vs Game/Work Score")
    plt.tight_layout()
    plt.savefig("price_vs_scores.png")
    plt.close()


def main():
    """Main function to run analysis."""
    df = load_data("final_pc_builds.csv")
    describe_prices(df)
    plot_price_distribution(df)
    count_builds_by_price(df)
    check_missing_values(df)
    show_correlation(df)
    plot_price_score_relation(df)


if __name__ == "__main__":
    main()
