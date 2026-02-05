import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_DIR = "Bell_paper_data/dataset_crop_detector"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tf)
val_dl = DataLoader(val_ds, batch_size=32)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("models/crop_detector.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in val_dl:
        x = x.to(DEVICE)
        out = model(x)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.numpy())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("Validation Accuracy:", round(acc * 100, 2), "%")
print("Confusion Matrix:\n", cm)
print("Class mapping:", val_ds.class_to_idx)
