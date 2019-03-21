def plot_training_curve(path, epochsize):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss. Requires Epoch size for plot size

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    import numpy as np

    epochsize = epochsize + 1
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    plt.plot(range(1, epochsize), train_err, label="Train")
    plt.plot(range(1, epochsize), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1, epochsize), train_loss, label="Train")
    plt.plot(range(1, epochsize), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()