import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from dataset import get_data_loaders
from model import DigitCNN

def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    misclassified_images, misclassified_preds, misclassified_targets = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            incorrect_mask = (preds != labels)
            if incorrect_mask.any():
                for img, pred, target in zip(images[incorrect_mask], preds[incorrect_mask], labels[incorrect_mask]):
                    if len(misclassified_images) < 10:
                        misclassified_images.append(img.cpu())
                        misclassified_preds.append(pred.item())
                        misclassified_targets.append(target.item())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 1. Classification Metrics
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*50)
    target_names = [f"Digit {i}" for i in range(10)]
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    # 3. Misclassified Examples
    if misclassified_images:
        num_examples = min(8, len(misclassified_images))
        fig, axes = plt.subplots(1, num_examples, figsize=(14, 2.5))
        for i in range(num_examples):
            img = misclassified_images[i].squeeze().numpy() * 0.3081 + 0.1307
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"P: {misclassified_preds[i]} | T: {misclassified_targets[i]}", color='red')
            axes[i].axis('off')
        plt.suptitle("Misclassified Digits (P = Predicted, T = True Target)")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_data_loaders(batch_size=64)

    model = DigitCNN().to(device)
    model.load_state_dict(torch.load("best_mnist_cnn.pth", map_location=device))
    evaluate_model(model, test_loader, device=device)