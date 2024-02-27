import copy
import math
import pickle
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import skew, kurtosis, median_abs_deviation, moment
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def rmsValue(array):
    n = len(array)
    squre = 0.0
    root = 0.0
    mean = 0.0

    # calculating Squre
    for i in range(0, n):
        squre += (array[i] ** 2)
    # Calculating Mean
    mean = (squre / float(n))
    # Calculating Root
    root = math.sqrt(mean)
    return root


def peak_to_rms_1D(array):
    return np.max(np.abs(array)) / rmsValue(array)


def signal_range_1D(array):
    return np.max(array) - np.min(array)


def stats_from_WD(s_signal, wv_type, decomp_lvl):
    # s_signal = normalize_data(s_signal)
    coeffs = pywt.wavedec(s_signal, wv_type, level=decomp_lvl)
    # cA, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    ft_vec = []

    for comp in coeffs:
        ft_vec.append([np.mean(comp), np.var(comp), np.std(comp), rmsValue(comp),
                       skew(comp), kurtosis(comp), peak_to_rms_1D(comp), rmsValue(comp) / np.mean(np.abs(comp)),
                       median_abs_deviation(comp), moment(comp, moment=2), np.min(comp),
                       np.max(comp), signal_range_1D(comp)])

    ft_vec = np.reshape(ft_vec, (len(coeffs) * len(ft_vec[0])), )  # 1,52
    return ft_vec


def get_train_ft(train_data, dc_type, dc_lvl):
    train_features = []
    for i in range(len(train_data)):
        ft_vec = stats_from_WD(train_data[i, :], dc_type, dc_lvl)
        train_features.append(ft_vec)

    train_features = np.array(train_features)

    return train_features


train_data = pd.read_csv('Epileptic Seizure Recognition.csv')
train_labels = train_data.iloc[:, -1]
train_labels = np.array(train_labels, dtype=np.int16)

feat_type = "DWT"
standard_scaler = True

if feat_type == "Time":
    train_data = train_data.iloc[:, 1:-1]
    train_data = np.array(train_data)
    # train_data = sc.fit_transform(train_data)
elif feat_type == "DWT":
    dwt_base = "db4"
    dwt_dc_level = 3
    train_data = train_data.iloc[:, 1:-1]
    train_data = np.array(train_data)
    print("Extracting DWT-based features\n")
    train_data = get_train_ft(train_data, dwt_base, dwt_dc_level)

###################################################################
# TRANSFORM TO A BINARY PROBLEM
train_labels[train_labels != 1] = 0  # 1 - seizure, 0 - no seizure
n_lab = np.unique(train_labels)
train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.33,
                                                                    random_state=42)

if standard_scaler:
    sc = StandardScaler()
    train_data = sc.fit_transform(train_data)
    test_data = sc.transform(test_data)

counts = []
for i in range(len(n_lab)):
    counts.append(np.sum(train_labels == i))
counts = counts / sum(counts)

print(f"Train data labels distributions: {counts * 100}\n")


class MyDNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(MyDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# Convert numpy arrays to PyTorch tensors
train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

# Creating DataLoader for training and validation sets
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
input_size = train_data.shape[1]  # 187  # Number of features
hidden_size1 = 40
hidden_size2 = 20
num_classes = len(n_lab)  # Number of classes
model = MyDNN(input_size, hidden_size1, hidden_size2, num_classes)

# WEIGHT THE LOSS DIFFERENTLY
# BECAUSE OF THE CLASS IMBALANCE
# (THIS VALUE CAN BE TUNED)
n_weights = torch.tensor([1.0, 5.0])
criterion = nn.CrossEntropyLoss(weight=n_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Print training loss after each epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
# Create DataLoader for the test set
test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Evaluate the model on test data
model.eval()
predicted_labels = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

np.set_printoptions(precision=4)

cm_norm = np.array(copy.copy(conf_matrix), dtype=float)
for i in range(0, num_classes):
    cm_norm[i, :] = cm_norm[i, :] / int(np.sum(cm_norm, axis=1)[i])

print("Confusion Matrix Normalized:")
print(cm_norm)

print(classification_report(true_labels, predicted_labels))


# Save the model to a file
model_filename = 'my_dnn_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved as {model_filename}")
