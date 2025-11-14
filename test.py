import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Frame
import matplotlib.image as mpimg

# ----------------------- DEVICE SETUP -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- DATASET SETUP -----------------------
images_path = []
labels = []
dir_path = "data"  # Change to your dataset path


for i in os.listdir(dir_path):

    for lable in os.listdir(f"{dir_path}/{i}"):
   
        for image in os.listdir(f"{dir_path}/{i}/{lable}"):

            images_path.append(f"{dir_path}/{i}/{lable}/{image}") # append path for each image
            labels.append((lable)) # append lable for each image
            
         

data_df = pd.DataFrame(list(zip(images_path, labels)), columns=["image_path", "labels"])

# Train, validation, test split
train_df = data_df.sample(frac=0.7, random_state=42)
temp_df = data_df.drop(train_df.index)
val_df = temp_df.sample(frac=0.5, random_state=42)
test_df = temp_df.drop(val_df.index)

# Label encoder
label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

# ----------------------- DATASET CLASS -----------------------
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(df["labels"]), dtype=torch.long).to(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image).to(device)
        return image, label

train_dataset = ImageDataset(train_df, transform)
val_dataset = ImageDataset(val_df, transform)
test_dataset = ImageDataset(test_df, transform)

# ----------------------- MODEL DEFINITION -----------------------
class ImageModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ImageModel, self).__init__()

        f1=32
        f2=64
        f3=128
        input_size = 64
        
        # Feature extractor
        self.features = nn.Sequential(
         
            # Block 1
            nn.Conv2d(3, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Block 2
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3, f3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # Automatically calculate flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            flatten_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ----------------------- TRAIN OR LOAD MODEL -----------------------
choice = input("Do you want to Train a new model or no? y/n ")

num_classes = data_df["labels"].nunique()
model = ImageModel(num_classes).to(device)

trained_new_model = False  # Flag to know if we trained a model

if choice.strip().lower() == "y":
    trained_new_model = True
    # Training setup
    LR = 0.0007
    BATCH_SIZE = 64
    EPOCHS = 7

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # For plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    #summary(model, (3, 128, 128))

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss_train, total_acc_train = 0, 0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            total_acc_train += (outputs.argmax(1) == lbls).sum().item()

        model.eval()
        total_loss_val, total_acc_val = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                total_loss_val += loss.item()
                total_acc_val += (outputs.argmax(1) == lbls).sum().item()

        # Save metrics for plotting
        train_losses.append(total_loss_train/len(train_loader))
        val_losses.append(total_loss_val/len(val_loader))
        train_accs.append(total_acc_train/len(train_dataset))
        val_accs.append(total_acc_val/len(val_dataset))

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]*100:.2f}% "
              f"| Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]*100:.2f}%")

    # Save the trained model
    torch.save(model, "saved_model_complete.pth")
    print("Model trained and saved as 'saved_model_complete.pth'.")

# If user chose pre-trained
else:
    model = torch.load("saved_model_complete.pth")
    model.eval()
    print("Loaded pre-trained model.")

# ----------------------- IMAGE PREDICTION -----------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        pred = torch.argmax(output, 1).item()
    return label_encoder.inverse_transform([pred])[0]

# ----------------------- GUI FOR IMAGE CHOOSING -----------------------
def choose_image():
    image_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if image_path:
        try:
            prediction = predict_image(image_path)
            img = mpimg.imread(image_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Prediction: {prediction}", fontsize=14, fontweight='bold', color='darkblue')
            plt.axis('off')
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

# ----------------------- GUI -----------------------
root = tk.Tk()
root.title("Image Classification")
root.geometry("400x200")
root.configure(bg="#f0f8ff")

frame = Frame(root, bg="#add8e6", padx=20, pady=20, relief="ridge", borderwidth=5)
frame.pack(pady=30)

label = Label(frame, text="Choose an image to classify", font=("Helvetica", 14), bg="#add8e6", fg="#00008b")
label.pack(pady=10)

button = Button(frame, text="Browse", command=choose_image, font=("Helvetica", 12), bg="#4682b4", fg="white",
                activebackground="#5f9ea0", activeforeground="white")
button.pack(pady=10)

root.mainloop()

# ----------------------- PLOTTING LOSS AND ACCURACY -----------------------
if trained_new_model:
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(train_losses, label="Train Loss")
    ax[0].plot(val_losses, label="Val Loss")
    ax[0].set_title("Loss per Epoch")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot([a*100 for a in train_accs], label="Train Accuracy")
    ax[1].plot([a*100 for a in val_accs], label="Val Accuracy")
    ax[1].set_title("Accuracy per Epoch")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy (%)")
    ax[1].legend()

    plt.show()
