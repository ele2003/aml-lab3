import os

import torch
from torch import nn
import wandb

from data.tiny_imagenet import get_dataloaders
from eval import validate
from models.customnet import CustomNet
from utils.visualization import show_class_samples


def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()                 # azzera i gradienti accumulati
        outputs = model(inputs)               # forward pass (logits, B x 200)
        loss = criterion(outputs, targets)    # CrossEntropyLoss
        loss.backward()                       # backpropagation
        optimizer.step()                      # aggiorna i pesi

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


if __name__ == '__main__':
    config = {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'batch_size': 32,
        'epochs': 10,
    }
    wandb.init(project='aml-lab3', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_dataloaders(batch_size=config['batch_size'])

    # Visualizzazione: un esempio per 10 classi, loggato su wandb
    fig = show_class_samples(train_loader)
    wandb.log({'samples': wandb.Image(fig)})

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['momentum'])

    os.makedirs('checkpoints', exist_ok=True)
    best_acc = 0

    # Run the training process for {epochs} epochs
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, device)

        # At the end of each training iteration, perform a validation step
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc})

        # Best validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    print(f'Best validation accuracy: {best_acc:.2f}%')
    wandb.finish()
