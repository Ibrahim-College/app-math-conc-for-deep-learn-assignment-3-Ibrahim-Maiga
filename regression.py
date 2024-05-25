import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    model = nn.Linear(input_size, output_size)
    torch.nn.init.normal_(model.weight, mean=0.0, std=0.01)
    torch.nn.init.constant_(model.bias, 0)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    learning_rate = 0.001  # Adjusted for better convergence
    num_epochs = 50000  # Increased to allow more thorough training
    input_features = X.shape[1]
    output_features = y.shape[1] if len(y.shape) > 1 else 1

    model = create_linear_regression_model(input_features, output_features)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    previous_loss = float("inf")
    tolerance = 1e-6  # More stringent stopping condition

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        current_loss = loss.item()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {current_loss}")

        if abs(previous_loss - current_loss) < tolerance:
            print("Stopping early due to minimal loss improvement.")
            break

        previous_loss = current_loss

    return model, loss

# Example usage:
# X, y are expected to be torch tensors with appropriate shapes
# model, final_loss = fit_regression_model(X, y)
