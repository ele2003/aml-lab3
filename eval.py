import torch
from torch import nn

from data.tiny_imagenet import get_dataloaders
from models.customnet import CustomNet


# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)               # forward pass (logits, B x 200)
            loss = criterion(outputs, targets)    # solo calcolo della loss, niente backward

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, val_loader = get_dataloaders()

    model = CustomNet().to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))

    validate(model, val_loader, nn.CrossEntropyLoss(), device)
