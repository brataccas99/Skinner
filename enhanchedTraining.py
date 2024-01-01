import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from torch.optim import lr_scheduler
import tqdm
from sklearn.metrics import classification_report, confusion_matrix

def load_data(input_path, batch_size):
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(input_path, "train"), data_transforms)
    val_ds = datasets.ImageFolder(os.path.join(input_path, "val"), data_transforms)

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        "validation": torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4),
    }

    return dataloaders

def initialize_model(device):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

    for layer in model.fc.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    model.fc = nn.Sequential(
        nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 10), nn.Softmax(dim=1)
    ).to(device)

    return model

def save_epoch_weights(model, weights_dir, epoch):
    epoch_weights_path = os.path.join(weights_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), epoch_weights_path)

def train_one_epoch(model, criterion, optimizer, dataloader, device, phase):
    model.train() if phase == "train" else model.eval()

    count = 0
    running_loss = 0.0
    running_corrects = 0
    all_labels = []
    all_preds = []

    with tqdm.tqdm(dataloader, desc=f"{phase} Epoch") as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if phase == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            count += 1

            # Collect labels and predictions for evaluation metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / (count * dataloader.batch_size)
    epoch_acc = running_corrects.float() / (count * dataloader.batch_size)

    return epoch_loss, epoch_acc, all_labels, all_preds

def evaluate_model(model, dataloader, device):
    model.eval()

    count = 0
    all_labels = []
    all_preds = []

    with tqdm.tqdm(dataloader, desc="Evaluation") as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)

            preds = torch.argmax(outputs, dim=1)
            count += 1

            # Collect labels and predictions for evaluation metrics
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, loss_file_path, weights_dir):
    best_val_acc = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training Phase
        train_loss, train_acc, _, _ = train_one_epoch(model, criterion, optimizer, dataloaders["train"], device, "train")

        # Validation Phase
        val_loss, val_acc, _, _ = train_one_epoch(model, criterion, None, dataloaders["validation"], device, "validation")

        # Save the model weights after each epoch
        save_epoch_weights(model, weights_dir, epoch + 1)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()

        # Print classification report and confusion matrix
        all_labels, all_preds = evaluate_model(model, dataloaders["validation"], device)
        class_report = classification_report(all_labels, all_preds, zero_division=1)
        print(f"Classification Report:\n{class_report}")

        conf_matrix = confusion_matrix(all_labels, all_preds)
        print(f"Confusion Matrix:\n{conf_matrix}")

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"Validation loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

        # Save losses and accuracies to the file
        with open(loss_file_path, "a") as loss_file:
            loss_file.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        scheduler.step(val_acc)  # Move this line outside the validation phase

    return best_model_wts

def main():
    input_path = "./trainingDataset/Hair Diseases - Final"
    BATCH_SIZE = 7
    num_epochs = 40

    dataloaders = load_data(input_path, BATCH_SIZE)
    
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = initialize_model(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1, verbose=True)

    loss_file_path = "./losses/losses.csv"
    weights_dir = "./trained_model_weights"
    os.makedirs(weights_dir, exist_ok=True)

    train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, loss_file_path, weights_dir)

if __name__ == '__main__':
    main()
