# plot_graph function
import matplotlib.pyplot as plt
def plot_graph(values1, values2, rng, label1, label2):
    plt.plot(range(rng), values1, label=label1)
    plt.plot(range(rng), values2, label=label2)
    plt.legend()
    plt.grid()
    plt.show()
if __name__ == '__main__':
    plot_graph()    