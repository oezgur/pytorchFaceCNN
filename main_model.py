import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

model_file = 'model2.pth'

DATADIR = "./faces/Train"

TESTDIR = "./testing_data"

# Read the CSV file
df = pd.read_csv('./faces/train.csv')
label_dict = {'YOUNG': 0, 'MIDDLE': 1, 'OLD': 2}
classes = ['Young', 'Middle', 'Old']
data = []
labels = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 100  # Define a fixed image size

print(device)

for index, row in df.iterrows():
    img_name = row['ID']
    category = row['Class']

    img_path = os.path.join(DATADIR, img_name)

    if os.path.isfile(img_path):
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize the image
        data.append(resized_array)
        labels.append(label_dict[category])

# Convert lists to numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int64)

# Train-Validation ayrımı (np: np_array cinsi olanlar)
train_data_np, val_data_np, train_labels_np, val_labels_np = train_test_split(data, labels, test_size=0.2)

# Reshape to input data
train_data_np = train_data_np.reshape(-1,1,IMG_SIZE, IMG_SIZE)
val_data_np = val_data_np.reshape(-1,1,IMG_SIZE, IMG_SIZE)

# Convert numpy arrays to PyTorch tensors
train_data_t = torch.from_numpy(train_data_np)
val_data_t= torch.from_numpy(val_data_np)
train_labels_t = torch.from_numpy(train_labels_np)
val_labels_t = torch.from_numpy(val_labels_np)
print("Converted training and validation data to tensors.")

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Datasetleri 
# Training Dataset:
training_dataset = MyDataset(train_data_t, train_labels_t)
# Validation Dataset:
validation_dataset = MyDataset(val_data_t, val_labels_t)

# İki dataseti de yükleme:
dataloader_train = DataLoader(training_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(validation_dataset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self, output_size=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        if output_size is not None:
            self.fc1 = nn.Linear(output_size, 128)
            self.fc2 = nn.Linear(128, 3)
            #self.fc3 = nn.Linear(64, 3)
        else:
            self.fc1 = None
            self.fc2 = None
            #self.fc3 = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.fc1 is None:
            output_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(output_size, 128)
            self.fc2 = nn.Linear(128, 3)
            #self.fc3 = nn.Linear(64, 3)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x
    

num_epochs = 20
net = Net()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)

if os.path.isfile(model_file):
    state_dict = torch.load(model_file)
    output_size = state_dict['fc1.weight'].shape[1]
    net = Net(output_size)
    # load the state dict into the model
    net.load_state_dict(state_dict)
else:
    net = Net()
    # Train the network
    # Training phase
    print("Training")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader_train), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        # Validation phase
    print("Validating")
    with torch.no_grad():  # No need to track gradients
        val_loss = 0.0
        for i, data in enumerate(tqdm(dataloader_val), 0):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        print('Validation Loss: %.3f' % (val_loss / len(dataloader_val)))

print('Finished Training')
torch.save(net.state_dict(), model_file)


# Evaluate accuracy on training data
correct_train = 0
total_train = 0
with torch.no_grad():
    for data in dataloader_train:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

print('Accuracy of the network on the training images: %d %%' % (
    100 * correct_train / total_train))

# Evaluate accuracy on validation data
correct_val = 0
total_val = 0
with torch.no_grad():
    for data in dataloader_val:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()

print('Accuracy of the network on the validation images: %d %%' % (
    100 * correct_val / total_val))

print("DONE.")
