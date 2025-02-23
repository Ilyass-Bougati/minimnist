from model.Model import NN
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np
import csv
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


# hyper parameters
BACH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 20
HIDDEN_LAYER = 20

# Reading the dataset from csv
# Reading the data
dataset = []
with open("images/dataset.csv", encoding="utf-8-sig") as file:
    filereader = csv.reader(file)
    for row in filereader:
        number = []
        for num in row:
            number.append(float(num))
        dataset.append(np.array(number))

random.shuffle(dataset)


# Separating the data
test_dataset_p = 0.2
test_data_len = int(len(dataset) * test_dataset_p)
train_data_len = len(dataset) - test_data_len

train_images = np.array([img[:-1] for img in dataset[:train_data_len]])
train_labels = np.array([int(img[-1]) for img in dataset[:train_data_len]])
test_images = np.array([img[:-1] for img in dataset[train_data_len:]])
test_labels = np.array([int(img[-1]) for img in dataset[train_data_len:]])

# Creating tensors dataset 
train_dataset = TensorDataset(torch.tensor(train_images).float(), torch.tensor(train_labels))
test_dataset = TensorDataset(torch.tensor(test_images).float(), torch.tensor(test_labels))

# Determining the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BACH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BACH_SIZE, 
    shuffle=False
)

# Creating the model
model = NN(HIDDEN_LAYER).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

data = {
    "loss": [],
    "accuracy": []
}

for epoch in tqdm(range(EPOCHS)):
    model.train()
    epoch_loss = 0.0
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)
    data["loss"].append(loss.item())

    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            # value, index
            _, predictions = torch.max(outputs,1) # 1 is the dimension
            n_samples += labels.shape[0] # number of samples in the current batch
            n_correct += (predictions == labels).sum().item()  # number of correct predictions

        data["accuracy"].append(n_correct / n_samples)  # accuracy
    

torch.save(model, "model.pth")

plt.plot(data["loss"])
plt.plot(data["accuracy"])
plt.show()