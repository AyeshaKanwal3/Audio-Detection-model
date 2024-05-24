import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Custom dataset class
class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.filepaths = []
        self.labels = []
        for label in ["real", "ai_generated"]:
            label_dir = os.path.join(directory, label)
            for filename in os.listdir(label_dir):
                if filename.endswith(".wav"):
                    self.filepaths.append(os.path.join(label_dir, filename))
                    self.labels.append(0 if label == "real" else 1)
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        audio, _ = librosa.load(filepath, sr=22050)
        if self.transform:
            audio = self.transform(audio)
        audio = torch.FloatTensor(audio).unsqueeze(0)  # Add channel dimension
        return audio, label



def extract_features(audio):
    # Ensure audio is a PyTorch tensor
    audio = torch.tensor(audio)
    
    # Convert tensor to NumPy array and squeeze if necessary
    audio_np = audio.squeeze().numpy() if len(audio.shape) > 1 else audio.numpy()
    
    # Extract MFCC features from the audio signal
    mfccs = librosa.feature.mfcc(y=audio_np, sr=22050, n_mfcc=40)
    
    # Convert MFCC features to PyTorch tensor
    return torch.tensor(mfccs).unsqueeze(0).unsqueeze(0)

# Define your Tortoise model class if not already defined
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # Adjust based on input size
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model():
    train_dataset = AudioDataset("dataset/train", transform=extract_features)
    val_dataset = AudioDataset("dataset/val", transform=extract_features)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = AudioClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "audio_classifier.pth")

if __name__ == "__main__":
    train_model()
