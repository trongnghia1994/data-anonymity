import matplotlib.pyplot as plt
import matplotlib.markers as markers
x_labels = [5, 10, 15, 20, 25, 30]
algos = ['OKA', 'GCCG', 'M3AR', 'U-M3AR']
short_algos = ['OKA', 'GCCG', 'M3AR & U-M3AR']

elapsed_time = [
    [2096.34, 1192.93, 851.56, 710.75, 629.58, 565.14],
    [612.82, 653.20, 642.15, 568.00, 585.24, 560.58],
    [1357.37, 1396.02, 1387.70, 1404.23, 1421.85, 1415.14],
    [943.07, 1142.87, 1230.96, 1278.23, 1335.33, 1349.90],
]

LRP = [
    [100, 100, 100, 100, 100, 100],
    [44.44, 51.11, 62.22, 68.89, 80.00, 82.22],
    [0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0],
]

NRP = [
    [37.78, 62.22, 77.78, 88.89, 88.89, 88.89],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 15.6, 22.2],
    # [0, 0, 0, 0, 15.6, 22.2],
]

DRP = [
    [137.78, 162.22, 177.78, 188.89, 188.89, 188.89],
    [44.44, 51.11, 62.22, 68.89, 80.00, 82.22],
    [0, 0, 0, 0, 15.6, 22.2],
    # [0, 0, 0, 0, 15.6, 22.2],
]

CAVG = [
    [1.43, 1.15, 1.08, 1.04, 1.02, 1.003],
    [2.01, 1.54, 1.36, 1.34, 1.19, 1.14],
    [2.05, 1.64, 1.47, 1.37, 1.30, 1.24],
    # [2.05, 1.64, 1.47, 1.37, 1.30, 1.24],
]

def plot_elapsed_time():
    for data_by_algo in elapsed_time:
        plt.plot(x_labels, data_by_algo, marker='x')

    plt.xlabel('Parameter k')
    plt.ylabel('Elapsed time (seconds)')
    plt.legend(algos)
    plt.show()


def plot_lrp():
    for data_by_algo in LRP:
        plt.plot(x_labels, data_by_algo, marker='x')        

    plt.xlabel('Parameter k')
    plt.ylabel('LRP (%)')
    plt.legend(short_algos)
    plt.show()


def plot_nrp():
    for data_by_algo in NRP:
        plt.plot(x_labels, data_by_algo, marker='x')        

    plt.xlabel('Parameter k')
    plt.ylabel('NRP (%)')
    plt.legend(short_algos)
    plt.show()


def plot_drp():
    for data_by_algo in DRP:
        plt.plot(x_labels, data_by_algo, marker='x')        

    plt.xlabel('Parameter k')
    plt.ylabel('DRP (%)')
    plt.legend(short_algos)
    plt.show()


# for data_by_algo in CAVG:
#     plt.plot(x_labels, data_by_algo, marker='x')
#     plt.ylabel('CAVG')


def plot_cavg():
    for data_by_algo in CAVG:
        plt.plot(x_labels, data_by_algo, marker='x')        

    plt.xlabel('Parameter k')
    plt.ylabel('CAVG')
    plt.legend(short_algos)
    plt.show()


# plot_elapsed_time()
# plot_lrp()
# plot_nrp()
# plot_drp()
plot_cavg()
