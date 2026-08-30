import argparse
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Set PyTorch MPS fallback for Apple Silicon compatibility
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ==============================================================================
# 1. MODEL ARCHITECTURE DEFINITION (Must MATCH training structure exactly)
# ==============================================================================
class TinyVGG(nn.Module):
    """
    TinyVGG Architecture matching the trained checkpoint.
    Includes BatchNorm2d and Dropout layers.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        # Block 1
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Block 2
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
# 2. INFERENCE FUNCTION
# ==============================================================================
def predict_image(image_path: str, model_path: str, class_names: list):
    # Hardware device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Define identical validation/inference transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Load and preprocess target image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found.")
        return

    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint file '{model_path}' not found. Please train the model first.")
        return

    raw_image = Image.open(image_path).convert("RGB")
    
    # Apply transforms and add Batch dimension: [3, 64, 64] -> [1, 3, 64, 64]
    image_tensor = transform(raw_image).unsqueeze(dim=0).to(device)

    # Re-instantiate model architecture & load state dictionary
    model = TinyVGG(
        input_shape=3,
        hidden_units=16,
        output_shape=len(class_names)
    ).to(device)
    
    model.load_state_dict(torch.load(f=model_path, map_location=device))

    # Run Evaluation Mode (Disables Dropout and uses running BatchNorm statistics)
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_prob, pred_class_idx = torch.max(probabilities, dim=1)
        
        predicted_label = class_names[pred_class_idx.item()]
        confidence_pct = pred_prob.item() * 100

    # Display Results
    print("\n" + "=" * 55)
    print("🔮 INFERENCE RESULT")
    print("=" * 55)
    print(f"🖼️ Target Image:     {image_path}")
    print(f"🏆 Top Prediction:   {predicted_label.upper()} ({confidence_pct:.1f}% confidence)")
    print("-" * 55)
    print("📊 Probability Breakdown:")
    for idx, class_name in enumerate(class_names):
        prob = probabilities[0][idx].item() * 100
        bar = "█" * int(prob / 5)  # Visual bar graph
        print(f"  • {class_name:<12}: {prob:5.1f}% | {bar}")
    print("=" * 55 + "\n")


# ==============================================================================
# 3. CLI ENTRY POINT
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Predict single image class using trained TinyVGG weights.")
    parser.add_argument("image_path", type=str, help="Path to the target image file (e.g., test.jpg).")
    parser.add_argument("--model-path", type=str, default="best_tinyvgg_model.pth", help="Path to trained .pth checkpoint.")
    # 🔑 修改：如果后续想测试自定义 4 分类数据集，才需要手动传入 --custom-dataset 标记
    parser.add_argument("--custom-dataset", action="store_true", help="Use custom 4-class labels ['bike', 'bus', 'car', 'plane'] instead of CIFAR-10.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 🔑 默认直接使用 CIFAR-10 的 10 种类别
    if not args.custom_dataset:
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        # 仅当显式指定 --custom-dataset 时才使用自定义类别
        classes = ['bike', 'bus', 'car', 'plane']

    predict_image(
        image_path=args.image_path,
        model_path=args.model_path,
        class_names=classes
    )
