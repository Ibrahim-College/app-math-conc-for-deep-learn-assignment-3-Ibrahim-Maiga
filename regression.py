import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.01  # Define the learning rate
    num_epochs = 10000  # Define the number of epochs
    input_features = X.shape[1]  # Extract the number of features from the input shape of X
    output_features = y.shape[1] if y.ndimension() > 1 else 1  # Extract the number of features from the output shape of y
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss()  # Use mean squared error loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previos_loss = float("inf")

    for epoch in range(num_epochs):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        
        if abs(previos_loss - loss.item()) < 1e-6:  # Stop the training when the loss is not changing much
            print(f"Stopping early at epoch {epoch}, loss {loss.item()}")
            break
        
        previos_loss = loss.item()

        if epoch % 1000 == 0:  # Print the loss every 1000 epochs
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, loss
