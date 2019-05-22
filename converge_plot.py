import pandas as pd
import matplotlib.pyplot as plt

def main():
    chord_df = pd.read_csv("out/leakage_comparison_chord_ratio.txt")
    plt.scatter(chord_df['Ratio'], chord_df['Reflection1'], label="Material 1")
    plt.scatter(chord_df['Ratio'], chord_df['Reflection2'], label="Material 2")
    plt.xlabel("Chord 1 / Chord 2")
    plt.ylabel("Reflection")
    plt.grid(b=True, which="both", axis="both")
    plt.legend(loc="best")
    plt.ylim(bottom=0.0, top=0.1)
    plt.xscale("log")
    plt.savefig("img/chord_ratio.png")
    plt.cla()
    plt.clf()
    thickness_df = pd.read_csv("out/leakage_comparison_thickness.txt")
    plt.scatter(thickness_df['Thickness'], thickness_df['Reflection1'], label="Material 1")
    plt.scatter(thickness_df['Thickness'], thickness_df['Reflection2'], label="Material 2")
    plt.xlabel("Thickness (cm)")
    plt.ylabel("Reflection")
    plt.grid(b=True, which="both", axis="both")
    plt.legend(loc="best")
    plt.xscale("log")
    plt.savefig("img/thickness.png")
    plt.cla()
    plt.clf()

if __name__ == '__main__':
    main()
