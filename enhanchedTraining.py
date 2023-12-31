import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from torch.optim import lr_scheduler
import tqdm
from sklearn.metrics import classification_report, confusion_matrix


# Define data transformations
data_transforms = transforms.Compose(
    [transforms.Resize((512, 512)), transforms.ToTensor()]
)

# Load the dataset
input_path = "./trainingDataset/Hair Diseases - Final"
BATCH_SIZE = 8
train_ds = datasets.ImageFolder(os.path.join(input_path, "train"), data_transforms)
val_ds = datasets.ImageFolder(os.path.join(input_path, "val"), data_transforms)

dataloaders = {
    "train": torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
    "validation": torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=True
    ),
}

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

# Replace the last fully-connected layer
model.fc = nn.Sequential(
    nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 10), nn.Softmax(dim=1)
).to(device)

# Weight initialization
nn.init.kaiming_normal_(model.fc[0].weight)
nn.init.zeros_(model.fc[0].bias)
nn.init.kaiming_normal_(model.fc[2].weight)
nn.init.zeros_(model.fc[2].bias)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

# Learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.1, verbose=True
)

# Create a file to save training and validation losses
loss_file_path = "./losses/losses.csv"

with open(loss_file_path, "w") as loss_file:
    loss_file.write("epoch,train_loss,train_acc,validation_loss,validation_acc\n")

# Training loop
best_val_acc = 0.0
best_model_wts = None


def train_model_with_scheduler(
    model, criterion, optimizer, scheduler, num_epochs, loss_file_path
):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0

        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            count = 0
            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm.tqdm(dataloaders[phase]):
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

            if phase == "validation":
                scheduler.step(val_acc)

                # Save the best model based on validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_wts = model.state_dict()

                # Print classification report and confusion matrix
                class_report = classification_report(all_labels, all_preds)
                print(f"Classification Report:\n{class_report}")

                conf_matrix = confusion_matrix(all_labels, all_preds)
                print(f"Confusion Matrix:\n{conf_matrix}")

            epoch_loss = running_loss / (count * BATCH_SIZE)
            epoch_acc = running_corrects.float() / (count * BATCH_SIZE)

            if phase == "train":
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc

            print(f"{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

            # Save losses and accuracies to the file
            with open(loss_file_path, "a") as loss_file:
                loss_file.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n"
                )

    return best_model_wts


# Train the model with the scheduler
best_model_weights = train_model_with_scheduler(
    model, criterion, optimizer, scheduler, num_epochs=40, loss_file_path=loss_file_path
)

# Save the best model weights
torch.save(best_model_weights, "./trained model weights")
