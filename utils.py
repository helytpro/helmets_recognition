import torch
import matplotlib
import matplotlib.pyplot as plt
import json

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f"model_pretrained_{pretrained}.pth") # мб слэш убирать не надо


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"accuracy_pretrained_{pretrained}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"loss_pretrained_{pretrained}.png")


def save_metrics_plot(precision, recall, pretrained):

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        precision, color='blue', linestyle='-', 
        label='Precision'
    )
    plt.xlabel('Epochs')
    plt.ylabel('precision')
    plt.savefig(f"precision_pretrained_{pretrained}.png")
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        recall, color='red', linestyle='-', 
        label='Recall'
    )
    plt.xlabel('Epochs')
    plt.ylabel('recall')
    plt.legend()
    plt.savefig(f"Recall_pretrained_{pretrained}.png")


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def recall(outputs, labels):
    _, preds = torch.max(outputs, 1)
    true_positives = (preds * labels).sum().item()
    false_negatives = ((1 - preds) * labels).sum().item()
    return true_positives / (true_positives + false_negatives)


def precision(outputs, labels):
    _, preds = torch.max(outputs, 1)
    true_positives = (preds * labels).sum().item()
    false_positives = (preds * (1 - labels)).sum().item()
    return true_positives / (true_positives + false_positives)


class metrics_calculation():
    def __init__(self, targets, predictions):
        self.classes = ['OFF', 'ON']
        self.accuracy = accuracy(targets, predictions)
        self.precision = recall(targets, predictions)
        self.recall = precision(targets, predictions)

    def print_metrics(self):
        print(f"Accuracy={self.accuracy:.2f}")
        for i, p in enumerate(self.precision):
            print(f"Precision for class {self.classes[i]}: {p:.2f}")
        for i, r in enumerate(self.recall):
            print(f"Recall for class {self.classes[i]}: {r:.2f}")

    def write_json(self):
        data = {}
        data["Accuracy"] = self.accuracy
        for i, p in enumerate(self.precision):
            data[f"Precision__{self.classes[i]}"] = p
        for i, r in enumerate(self.recall):
            data[f"Recall__{self.classes[i]}"] = r
        with open('metrics.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
