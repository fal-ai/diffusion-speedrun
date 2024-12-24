import torch
import torch.nn as nn
from safetensors.torch import safe_open

with safe_open("tokenize_dataset/imagenet_ci8x8.safetensors", framework="pt") as f:
    data = f.get_tensor("latents")
    labels = f.get_tensor("labels").long()

print(data.shape)  # 1281167, 16, 32, 32
print(labels.shape)  # 1281167

data = data.reshape(data.shape[0], -1)  # Flatten to (N, 16*32*32)
gen = torch.Generator()
gen.manual_seed(42)
perm = torch.randperm(data.size(0), generator=gen)
data = data[perm]
labels = labels[perm]

train_size = int(0.95 * len(data))

train_data = data[:train_size]
train_labels = labels[:train_size]
val_data = data[train_size:]
val_labels = labels[train_size:]


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def forward(self, x):
        return self.linear(x)


input_dim = 16 * 32 * 32  # Flattened input dimension
output_dim = 1000  # ImageNet has 1000 classes
model = LinearRegression(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 4096
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    for i in range(0, len(train_data), batch_size):
        batch_data = (
            train_data[i : i + batch_size].to(torch.float32).to(device) * 16.0 / 255.0
        )

        batch_labels = train_labels[i : i + batch_size].to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            batch_data = (
                val_data[i : i + batch_size].to(torch.float32).to(device) * 16.0 / 255.0
            )
            batch_labels = val_labels[i : i + batch_size].to(device)

            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%"
    )

print("Training finished!")
