"""Analyze PC build price and performance data."""
import pandas as pd
import matplotlib.pyplot as plt
from config.settings import DATASET_PATH

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
    plt.savefig("data/price_distribution.png")
    plt.close()


def count_builds_by_price(df):
    """Print the number of builds in predefined price ranges."""
    bins = [0, 1000, 1500, 2500, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
    grouped = pd.cut(df["Total Price"], bins=bins)
    max_price = df["Total Price"].max()
    min_price = df["Total Price"].min()
    print("Number of builds by price range:")
    print(df.groupby(grouped).size())
    print(f"Maximum Total Price: {max_price}")
    print(f"Minimum Total Price: {min_price}")


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
    plt.savefig("data/price_vs_scores.png")
    plt.close()

def show_max_min_scores(df):
    """Print maximum and minimum scores for Game and Work."""
    max_game_score = df["Game Score"].max()
    max_work_score = df["Work Score"].max()
    min_game_score = df["Game Score"].min()
    min_work_score = df["Work Score"].min()
    print(f"\nMaximum Game Score: {max_game_score}")
    print(f"Maximum Work Score: {max_work_score}")
    print(f"Minimum Game Score: {min_game_score}")
    print(f"Minimum Work Score: {min_work_score}")

def analyze_gpu_performance(df):
    """Analyze performance of a specific GPU."""
    # Цільові GPU
    target_gpus = ["rx 9070 xt", "rtx 5080", "rtx 5090"]
    for gpu in target_gpus:
        gpu_df = df[df["Video Card"].str.contains(gpu, case=False)]
        if gpu_df.empty:
            print(f"No builds found for GPU: {gpu}")
            continue

        print(f"\n🎮 {gpu.upper()}")
        print(f"Кількість збірок: {len(gpu_df)}")
        print(f"Середній Game Score: {gpu_df['Game Score'].mean():.2f}")
        print(f"Середній Work Score: {gpu_df['Work Score'].mean():.2f}")
        print(f"Макс. Game Score: {gpu_df['Game Score'].max():.2f}")
        print(f"Макс. Work Score: {gpu_df['Work Score'].max():.2f}")
        print(f"Середня ціна: {gpu_df['Total Price'].mean():.2f}$")
        print(f"Макс. ціна: {gpu_df['Total Price'].max():.2f}$")
        print(f"Мін. ціна: {gpu_df['Total Price'].min():.2f}$")

    gpu_stats = df.groupby("Video Card").agg(
        count=("Video Card", "count"),
        avg_game_score=("Game Score", "mean"),
        avg_work_score=("Work Score", "mean"),
        avg_price=("Total Price", "mean")
    ).sort_values("avg_game_score", ascending=False)
    print("\n📊 Статистика по відеокартам:")
    print(gpu_stats[0:29])

def analyze_top_builds(df):
    """Analyze top builds based on price."""
    # Фільтрація по ціні
    min_price = 6000
    max_price = 10600
    filtered_df = df[(df['Total Price'] >= min_price) & (df['Total Price'] <= max_price) & (df['Game Score'] > 170) & (df['Work Score'] > 170) & (df['CPU'].str.contains("9700x", case=False))]

    # Підрахунок збірок з кожним CPU
    cpu_counts = filtered_df['CPU'].value_counts()

    # Вивід результатів
    print(f"Знайдено {len(filtered_df)} збірок у ціновому діапазоні {min_price}-{max_price}:\n")
    print(filtered_df)

    print("\nКількість збірок з кожним CPU у цьому діапазоні:\n")
    print(cpu_counts)

def main():
    """Main function to run analysis."""
    df = load_data(DATASET_PATH)
    describe_prices(df)
    plot_price_distribution(df)
    count_builds_by_price(df)
    check_missing_values(df)
    show_correlation(df)
    plot_price_score_relation(df)
    show_max_min_scores(df)
    analyze_gpu_performance(df)
    analyze_top_builds(df)

if __name__ == "__main__":
    main()
