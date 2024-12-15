import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data

class ThreeLayerNNTorch:
    def __init__(self, train_X,train_y,test_X,test_y):
        train_dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        activations = {'tanh': (nn.Tanh, 'xavier'), 'relu': (nn.ReLU, 'he')}
        layer_depths = [3, 5, 9]
        layer_widths = [5, 10, 25, 50, 100]

        all_losses = {}


        for act_name, (activation, init_method) in activations.items():
            for depth in layer_depths:
                for width in layer_widths:
                    config_label = f"{act_name}-depth{depth}-width{width}"
                    print(f"Training model with {act_name}, depth={depth}, width={width}")
                    hidden_layers = [width] * depth
                    model = self.create_model(train_X.shape[1], hidden_layers, 2, activation, init_method)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    losses = self.train_model(model, train_loader, criterion, optimizer)
                    accuracy = self.evaluate_model(model, train_loader)
                    print(f"Train Error: {1-accuracy:.4f}\n")
                    accuracy = self.evaluate_model(model, test_loader)
                    print(f"Test Error: {1-accuracy:.4f}\n")
                    all_losses[config_label] = losses

        self.plot_all_losses(all_losses)


    def create_model(self,input_size, hidden_layers, output_size, activation, init_method):
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_layers:
            layer = nn.Linear(current_size, hidden_size)
            if init_method == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif init_method == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            layers.append(layer)
            layers.append(activation())
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        model = nn.Sequential(*layers)
        return model

    def train_model(self,model, data_loader, criterion, optimizer, epochs=20):
        model.train()
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            losses.append(avg_loss)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return losses

    def evaluate_model(self,model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        return correct / total

    def plot_all_losses(self,loss_data):
        plt.figure(figsize=(12, 8))
        for label, losses in loss_data.items():
            plt.plot(range(1, len(losses) + 1), losses, marker='o', label=label)
        plt.title('Training Loss per Epoch for All Configurations')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()
