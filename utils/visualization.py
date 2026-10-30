import matplotlib.pyplot as plt
import numpy as np

from data.tiny_imagenet import MEAN, STD


def denormalize(image):
    """Function to denormalize image for visualization"""
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    image = image * np.array(STD) + np.array(MEAN)
    image = np.clip(image, 0, 1)
    return image


def show_class_samples(dataloader, n_classes=10):
    """Visualize one example for each class, for n_classes classes."""
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(15, 6))
    axes = axes.flatten()

    classes_sampled = set()
    found_classes = 0

    for inputs, classes in dataloader:
        for img, label in zip(inputs, classes):
            label_item = label.item()
            if label_item not in classes_sampled:
                classes_sampled.add(label_item)

                axes[found_classes].imshow(denormalize(img))
                axes[found_classes].set_title(f"Class: {label_item}")
                axes[found_classes].axis('off')

                found_classes += 1
                if found_classes >= n_classes:
                    break
        if found_classes >= n_classes:
            break

    plt.tight_layout()
    return fig


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    """Curve di loss e accuracy per epoca."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='train')
    ax1.plot(epochs, val_losses, label='val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

    ax2.plot(epochs, train_accs, label='train')
    ax2.plot(epochs, val_accs, label='val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend()

    plt.tight_layout()
    return fig
