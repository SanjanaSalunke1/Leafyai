import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

DATA_DIR = "Bell_paper_data/dataset_crop_detector"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tf)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=32)

model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    loss_sum = 0

    progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/5", leave=True)

    for x, y in progress:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} completed | Total Loss: {loss_sum:.3f}")

torch.save(model.state_dict(), "models/crop_detector.pth") 
print("Saved crop detector")