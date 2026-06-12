# Image Classification with Transfer Learning

Three notebooks demonstrating transfer learning:

| Notebook | Framework | Task |
|---|---|---|
| `pytorch_transfer_learning.ipynb` | Pure PyTorch | Damaged vs New Vehicles |
| `HumanvsAnimal.ipynb` | fast.ai | Human vs Animal |
| `Deep_learning.ipynb` | fast.ai | Forest vs Bird |

---

## Pipeline — `pytorch_transfer_learning.ipynb`

This notebook walks through every step manually with pure PyTorch:

```
Dataset (folder of images)
    ↓
ImageFolder (reads folders → assigns labels)
    ↓
DataLoader (batches + shuffles)
    ↓
Batch [32, 3, 224, 244]
    ↓
ResNet18 (pretrained, frozen)
    ↓
Features [32, 512]
    ↓
FC Layer (the ONLY thing trained)
    ↓
Logits [32, 2]
    ↓
Softmax → probabilities
    ↓
Prediction (argmax)
```

---

### Step by Step

#### 1. Load ResNet18 & Freeze It (cells 2–6)

```python
model = resnet18(weights=ResNet18_Weights.DEFAULT)   # cell 2
```

Load ResNet18 pretrained on ImageNet.

```python
for params in model.parameters():
    params.requires_grad = False                      # cell 4 — freeze everything
```

Stop gradients for all layers — they won't be updated during training.

```python
model.fc = nn.Linear(model.fc.in_features, 2)         # cell 5 — swap the head
```

Replace the original 1000-class classifier with a 2-class one. Only this new layer has `requires_grad = True`.

```python
# cell 6 — confirms only fc.weight and fc.bias will train
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)   # → fc.weight, fc.bias
```

---

#### 2. Download & Prepare Images (cells 7–8)

Images are downloaded via DuckDuckGo search into `DamagedvsBranded/Damaged Vehicles/` and `DamagedvsBranded/New Vehicles/`. Corrupted images are removed with `verify_images`.

---

#### 3. Create Dataset (cell 9 — THIS is the line that returns the dataset)

```python
dataset = datasets.ImageFolder(root="DamagedvsBranded", transform=transform)
dataset.classes, dataset.class_to_idx
```

`ImageFolder` reads the subfolder names as class labels:
- `Damaged Vehicles/` → index `0`
- `New Vehicles/` → index `1`

Output: `(['Damaged Vehicles', 'New Vehicles'], {'Damaged Vehicles': 0, 'New Vehicles': 1})`

---

#### 4. Train/Validation Split (cell 10)

```python
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
```

70% train, 30% validation. Output: `(133, 58, 191)` — 133 train, 58 val, 191 total.

---

#### 5. DataLoaders (cell 11)

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(valid_dataset, batch_size=32)
```

Wraps datasets into iterables. Each batch is a tuple `(images, labels)` where:
- `images.shape` = `[32, 3, 224, 244]`
- `labels.shape` = `[32]`

---

#### 6. Loss & Optimizer (cells 12–13)

```python
criterion = nn.CrossEntropyLoss()                    # cell 12
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # cell 13 — only FC layer
```

---

#### 7. Training Loop (cell 17)

```python
for epoch in range(20):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)          # forward → [32, 2] logits
        loss = criterion(outputs, labels)  # CrossEntropyLoss
        loss.backward()                  # backprop
        optimizer.step()                 # update fc.weight, fc.bias
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
```

Loss drops from ~0.56 to ~0.20 over 20 epochs.

---

#### 8. Inference on a New Image (cell 31 — output: prediction + image)

```python
input_path = "./bmw.jpg"
image = Image.open(input_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)          # → [1, 3, 224, 224]

model.eval()
with torch.no_grad():
    outputs = model(input_tensor)                      # logits

prob = F.softmax(outputs, dim=1)                       # probabilities
pred = outputs.argmax()                                # predicted class

class_names = ["Damaged Vehicles", "New Vehicles"]
print(class_names[pred.item()], prob)                  # → "New Vehicles [[0.016, 0.984]]"

im = Image.open(input_path)
im.to_thumb(256, 256)                                  # shows the car image
```
---
### Output

<p align="center">
  <img width="400" src="https://github.com/user-attachments/assets/d2d16a92-2b67-46e1-9dda-cd2c47b25dd2" alt="Prediction text output">
  <img width="400" src="https://github.com/user-attachments/assets/bff876af-69fe-4ed8-be73-32a601448965" alt="Image thumbnail output">
</p>

---

## fast.ai Notebooks

`HumanvsAnimal.ipynb` and `Deep_learning.ipynb` use fast.ai's higher-level API which wraps the same PyTorch pipeline:

```python
dls = DataBlock(...).dataloaders(path, bs=32)          # ImageFolder + DataLoader
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)                                    # trains with gradual unfreezing

is_human, pred_idx, probs = learn.predict(PILImage.create('photo.jpg'))
```

The `Deep_learning.ipynb` also contains a **markdown explainer on transfer learning** — covering feature extractor vs fine-tuning vs partial fine-tuning, model selection, and the modern foundation-model approach.

---

## Setup

```bash
uv sync
```

Dependencies: PyTorch, torchvision, fastai, fastcore, fastdownload, duckduckgo-search.
