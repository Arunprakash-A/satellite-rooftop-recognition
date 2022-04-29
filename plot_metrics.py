from matplotlib import pyplot as plt
import torch

def plot():
    metrics = torch.load('./chkpoints/metrics')
    loss = metrics['Loss']
    accuracy = metrics['accuracy']
    plt.subplot(2,1,1)
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('BCELoss')
    plt.subplot(2,1,2)
    plt.plot(accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
if __name__ == '__main__':
    plot()
