import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, network_name, dir):
        self.network_name = network_name
        self.dir = dir

    def plot_average_acc_line_graph(self, accuracy_list, val_accuracy_list):
        average_accuracy_list = np.average(accuracy_list, axis=0)
        average_val_accuracy_list = np.average(val_accuracy_list, axis=0)
        std_dev_acc_list = np.std(accuracy_list, axis=0)
        std_dev_val_acc_list = np.std(val_accuracy_list, axis=0)

        epochs = range(1, len(average_accuracy_list) + 1)
        plt.plot(epochs, average_accuracy_list, 'r', label='Training')
        plt.fill_between(epochs, average_accuracy_list - std_dev_acc_list, average_accuracy_list + std_dev_acc_list,
                         color='r',
                         alpha=0.2)
        plt.plot(epochs, average_val_accuracy_list, 'g', label='Validation')
        plt.fill_between(epochs, average_val_accuracy_list - std_dev_val_acc_list,
                         average_val_accuracy_list + std_dev_val_acc_list, color='g', alpha=0.2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(np.arange(0, 35, 5))
        plt.yticks(np.arange(0, 0.5, 0.1))
        plt.title(self.network_name + ' Model Average Accuracy', fontsize=14)
        plt.legend(fontsize=10)
        plt.savefig(self.dir + self.network_name + '_Val Acc VS Acc Curves.png')
        plt.show()

    def plot_average_loss_line_graph(self, loss_list, val_loss_list):
        average_loss_list = np.average(loss_list, axis=0)
        average_val_loss_list = np.average(val_loss_list, axis=0)
        std_dev_loss_list = np.std(loss_list, axis=0)
        std_dev_val_loss_list = np.std(val_loss_list, axis=0)

        epochs = range(1, len(average_loss_list) + 1)
        plt.plot(epochs, average_loss_list, 'r', label='Training Loss')
        plt.fill_between(epochs, average_loss_list - std_dev_loss_list, average_loss_list + std_dev_loss_list,
                         color='r',
                         alpha=0.2)
        plt.plot(epochs, average_val_loss_list, 'g', label='Validation Loss')
        plt.fill_between(epochs, average_val_loss_list - std_dev_val_loss_list,
                         average_val_loss_list + std_dev_val_loss_list,
                         color='g', alpha=0.2)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.xticks(np.arange(0, 35, 5))
        plt.title(self.network_name + ' Model Loss', fontsize=14)
        plt.legend(fontsize=10)
        plt.savefig(self.dir + self.network_name + '_Val Loss VS Loss Curves.png')
        plt.show()

    def plot_bar_graph(self, network1_name, acc1, std1, network2_name, acc2, std2):
        name = [network1_name, network2_name]
        value = [acc1, acc2]
        error = [std1, std2]

        plt.bar(name, value, yerr=error, capsize=20)
        plt.xlabel('Network')
        plt.ylabel('Accuracy')
        plt.title(network1_name + ' accuracy vs ' + network2_name + ' accuracy', fontsize=14)
        plt.savefig(self.dir + 'Network Accuracy Comparison.png')
        plt.show()
