import torch
import torchvision.models as models
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

# ============ STEP 1: Load Pre-Trained Model ============
print("Loading pre-trained MobileNetV2...")
model = models.mobilenet_v2(pretrained=True)

# Freeze all layers (don't train them)
for param in model.features.parameters():
    param.requires_grad = False

# Replace last layer for binary classification (Fresh vs Rotten)
num_classes = 2
model.classifier[1] = nn.Linear(1280, num_classes)

# ============ STEP 2: Prepare Data ============
class FoodDataset(Dataset):
    def __init__(self, image_dir, labels_csv):
        self.image_dir = image_dir
        self.df = pd.read_csv(labels_csv)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.df.iloc[idx]['image'])
        label = self.df.iloc[idx]['label']  # 0=Fresh, 1=Rotten
        
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, label

print("Setting up datasets... (Note: Make sure labels_csv is prepared before running training)")
# Create datasets - Uncomment these when dataset is fully ready
# train_dataset = FoodDataset('../01_datasets/images/', '../01_datasets/train_labels.csv')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ============ STEP 3: Setup Training ============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# Only optimize the last layer (new)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ============ STEP 4: Train ============
def train_model():
    print("Starting training...")
    model.train()
    for epoch in range(10):  # Just 10 epochs needed!
        # for batch_idx, (images, labels) in enumerate(train_loader):
        #     images, labels = images.to(device), labels.to(device)
        #     
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     
        #     if batch_idx % 10 == 0:
        #         print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
        print(f"Epoch {epoch} simulated...")

# ============ STEP 5: Save Model ============
def save_model():
    os.makedirs('../02_models', exist_ok=True)
    torch.save(model.state_dict(), '../02_models/food_spoilage_model.pth')
    print("✓ Model saved to 02_models/food_spoilage_model.pth!")

# ============ STEP 6: Use for Prediction ============
def predict(image_path):
    model.eval()
    test_img = Image.open(image_path).convert('RGB')
    test_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])(test_img)

    with torch.no_grad():
        output = model(test_img.unsqueeze(0).to(device))
        probability = torch.nn.functional.softmax(output, dim=1)
        
        fresh_prob = probability[0][0].item()
        rotten_prob = probability[0][1].item()
        
        print(f"Fresh: {fresh_prob:.2%}")
        print(f"Rotten: {rotten_prob:.2%}")

if __name__ == "__main__":
    print("Transfer learning script initialized. Ready to train once dataset is downloaded and prepared.")
    # train_model()
    # save_model()
