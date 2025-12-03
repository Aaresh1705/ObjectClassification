from nms import non_max_suppression
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from process_crops import cropDataLoader
from model import get_vgg16_model

from copy import deepcopy

# We define the training as a function so we can easily re-use it.
def train(model, optimizer, trainset, train_loader, valset, val_loader, device: torch.device, num_epochs=10):
    def loss_fun(output, target):
        # return F.nll_loss(torch.log(output), target)
        return F.cross_entropy(output, target)

    out_dict = {'train_acc': [],
                'val_acc': [],
                'train_loss': [],
                'val_loss': []}
    best_acc = 0
    best_model = deepcopy(model)

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        # For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc='Training'):
            data, target = data.to(device), target.to(device)

            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = loss_fun(output, target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
        # Comput the val accuracy
        val_loss = []
        val_correct = 0
        model.eval()
        for data, target in tqdm(val_loader, total=len(val_loader), leave=False, desc='Validating'):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            val_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            val_correct += (target == predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct / len(trainset))
        out_dict['val_acc'].append(val_correct / len(valset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['val_loss'].append(np.mean(val_loss))
        tqdm.write(f"{epoch + 1}\t Loss train: {np.mean(train_loss):.3f}\t validation: {np.mean(val_loss):.3f}\t "
              f"Accuracy train: {out_dict['train_acc'][-1] * 100:.1f}%\t validation: {out_dict['val_acc'][-1] * 100:.1f}%")

        if out_dict['val_acc'][-1] > best_acc:
            best_acc = out_dict['val_acc'][-1]
            best_model = deepcopy(model)

    return out_dict, best_model

def plot_training(training_dict: dict):
    def default_plot(ax: plt.Axes):
        ax.legend()
        ax.grid(linestyle='--')
        ax.spines[['right', 'top']].set_visible(False)
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MultipleLocator(1))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(14)
    x = range(1, len(training_dict['train_acc']) + 1)

    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(x, training_dict['train_acc'], label='Train')
    ax1.plot(x, training_dict['val_acc'], label='Validation')
    default_plot(ax1)


    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(x, training_dict['train_loss'], label='Train')
    ax2.plot(x, training_dict['val_loss'], label='Validation')
    default_plot(ax2)

    fig.suptitle('Training results with Adam without data augmentations')
    fig.tight_layout()

    name = 'figures/val'
    plt.savefig(name + '.pdf')
    plt.savefig(name + '.png')
    plt.show()

def save_model(model):
    while True:
        save = input(f"Do you want to save the model? [y/n]\n")
        if save.lower() == 'y':
            break
        elif save.lower() == 'n':
            return
        else:
            print("Please enter y or n.")

    files = [file.split('\\')[-1] for file in glob.glob('models/*.pth')]
    print(f'Current model files:')
    for file in files:
        print(f'\t{file}')

    name = input('Enter file name: ')

    torch.save(model.state_dict(), f'models/{name}.pth')

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_vgg16_model(pretrained=True)
    model.to(device)

    transforms = [transforms.RandomRotation(30), transforms.RandomVerticalFlip(0.25), transforms.RandomHorizontalFlip(0.25)]
    (train_loader, val_loader, _), (trainset, valset, _) = cropDataLoader(batch_size=64, transform=transforms)

    optimizer = torch.optim.Adam(model.parameters())
    out_dict, best_model = train(model, optimizer, trainset, train_loader, valset, val_loader, device=device, num_epochs=10)

    plot_training(out_dict)

    save_model(best_model)