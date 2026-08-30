import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set PyTorch MPS fallback for Apple Silicon compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ==============================================================================
# EARLY STOPPING CLASS (Saves the best model based on Validation Loss)
# ==============================================================================
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience: int = 5, delta: float = 0.001, save_path: str = "best_tinyvgg_model.pth"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.delta:
            print(f"  🎯 Val Loss improved from {self.best_loss:.4f} to {val_loss:.4f}! Saving best model to '{self.save_path}'")
            self.best_loss = val_loss
            torch.save(obj=model.state_dict(), f=self.save_path)
            self.counter = 0  # Reset counter
        else:
            self.counter += 1
            print(f"  ⚠️ Val Loss did not improve ({self.best_loss:.4f}). EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ==============================================================================
# TINYVGG ARCHITECTURE DEFINITION
# ==============================================================================
class TinyVGG(nn.Module):
    """
    TinyVGG Architecture inspired by CNN Explainer (https://poloclub.github.io/cnn-explainer/).
    Includes BatchNorm2d and Dropout for training stability and performance.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        # Convolutional Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Convolutional Block 2
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# ==============================================================================
# COMMAND LINE ARGUMENT PARSER
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyVGG on CIFAR-10 or Custom Dataset.")
    
    # Dataset Mode Option (Default: CIFAR-10)
    parser.add_argument(
        "--custom-dataset",
        action="store_true",
        help="Use custom local dataset ('dataset/train' and 'dataset/val') instead of default CIFAR-10."
    )
    
    # Training Hyperparameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train (default: 30)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    
    return parser.parse_args()


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    print("▶️ Starting TinyVGG Training Pipeline...\n")

    MODEL_SAVE_PATH = "best_tinyvgg_model.pth"

    # --------------------------------------------------------------------------
    # STEP 1: HARDWARE SETUP
    # --------------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Acceleration Hardware: Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Acceleration Hardware: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("⚠️ Acceleration Hardware not found. Falling back to CPU")

    # --------------------------------------------------------------------------
    # STEP 2: DATA PREPROCESSING & LOADING
    # --------------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Dynamic Dataset Selection: CIFAR-10 is default
    if not args.custom_dataset:
        print("📦 Data Source: PyTorch Built-in Dataset (CIFAR-10) [Default]")
        train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
        val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=val_transform)
        class_names = train_data.classes
    else:
        print("📁 Data Source: Custom Local Directory ('dataset/train' & 'dataset/val')")
        train_dir = "dataset/train"
        val_dir = "dataset/val"

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError(
                f"\n❌ Custom dataset directories not found!\n"
                f"Please create '{train_dir}' and '{val_dir}' with subfolders per class.\n"
                f"Or remove '--custom-dataset' parameter to fall back to CIFAR-10."
            )

        train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_data = datasets.ImageFolder(root=val_dir, transform=val_transform)
        class_names = train_data.classes

    num_classes = len(class_names)
    print(f"📊 Dataset Info: {len(train_data)} training samples, {len(val_data)} validation samples across {num_classes} classes.")

    # Optimized num_workers for macOS stability
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --------------------------------------------------------------------------
    # STEP 3: MODEL INSTANTIATION & TENSOR PROBE
    # --------------------------------------------------------------------------
    model = TinyVGG(
        input_shape=3,
        hidden_units=16,
        output_shape=num_classes
    ).to(device)

    # Tensor shape probe
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    print("\n--- Tensor Dimension Inspection ---")
    with torch.no_grad():
        dummy_out = model(dummy_input)
        print(f"Input Shape: {dummy_input.shape} -> Output Shape: {dummy_out.shape}")
    print("-----------------------------------\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    early_stopper = EarlyStopping(patience=5, delta=0.001, save_path=MODEL_SAVE_PATH)

    # --------------------------------------------------------------------------
    # STEP 4: TRAINING & VALIDATION LOOP
    # --------------------------------------------------------------------------
    print(f"🧠 Training started on {device}...\n")

    for epoch in range(args.epochs):
        # --- Phase 1: Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # --- Phase 2: Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)

                val_outputs = model(X)
                v_loss = criterion(val_outputs, y)

                val_loss += v_loss.item() * X.size(0)
                _, val_preds = torch.max(val_outputs, dim=1)
                val_correct += (val_preds == y).sum().item()
                val_total += y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:02d}/{args.epochs:02d}] "
              f"| LR: {current_lr:.6f} "
              f"| Train Loss: {epoch_train_loss:.4f} (Acc: {epoch_train_acc:.1f}%) "
              f"| Val Loss: {epoch_val_loss:.4f} (Acc: {epoch_val_acc:.1f}%)")

        # --- Phase 3: Learning Rate Scheduler & Early Stopping ---
        scheduler.step(epoch_val_loss)
        early_stopper(val_loss=epoch_val_loss, model=model)

        if early_stopper.early_stop:
            print(f"\n🛑 Early Stopping triggered! Validation loss stopped improving for {early_stopper.patience} consecutive epochs.")
            break

    print(f"\n✨ Training pipeline finished. Best model checkpoint saved to '{MODEL_SAVE_PATH}'.")
