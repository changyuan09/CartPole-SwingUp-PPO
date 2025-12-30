import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load csv (expects columns: episode, avg_return, loss)
    df = pd.read_csv("train_log.csv")

    episodes = df["episode"]
    avg_return = df["avg_return"]
    loss = df["loss"]

    plt.figure(figsize=(10, 5))

    # Left y-axis: avg_return
    ax1 = plt.gca()
    ax1.plot(episodes, avg_return, label="Average Return", linewidth=1.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Average Return")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Right y-axis: loss
    ax2 = ax1.twinx()
    ax2.plot(episodes, loss, label="Loss", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Loss")

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    plt.title("Training Curve: Average Return & Loss vs Episode")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
