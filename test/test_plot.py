#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    data = pd.read_csv("test_out.csv")
    plt.plot(data['xvals'], data['yvals'], label="y(x) - Matrix Exponentiation", color='m')
    plt.plot(data['xvals'], data['uvals'], label="u(x) = y'(x) - Matrix Exponentiation", color='b')
    plt.plot(data['xvals'], data['ybench'], label="y(x) - Analytical", color='r', linestyle=':')
    plt.plot(data['xvals'], data['ubench'], label="u(x) = y'(x) - Analytical", color='c', linestyle=':')
    plt.legend(loc='best')
    plt.xlabel("x")
    plt.ylabel("Function Results")
    plt.title("Benchmark Test")
    plt.grid(b=True, which="both", axis="both")
    plt.tight_layout()
    plt.savefig("test_plot.png")
    plt.cla()
    plt.clf()

if __name__ == '__main__':
    main()
