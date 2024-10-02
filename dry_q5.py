import torch
import torch.nn as nn

# Create a single-channel 32x32 input image (x) and target output image (y)
x = torch.rand(1, 1, 32, 32, requires_grad=True)  # Input image
y = torch.rand(1, 1, 32, 32)  # Ground truth image

# Define a Conv2d layer with 1 input and 1 output channel, kernel size of 32 (same as the image size)
conv_layer = nn.Conv2d(1, 1, 32, bias=True)
print(f"W size: {conv_layer.weight.size()}")
print(f"b size: {conv_layer.bias.size()}")

# Forward pass through the convolutional layer
y_hat = conv_layer(x)
print(f"y_hat: {y_hat}, y_hat size: {y_hat.size()}")

# Compute the loss (sum of squared differences)
loss = ((y_hat - y) ** 2).sum()
print(f"loss: {loss}, loss size: {loss.size()}")

# Backward pass to compute gradients automatically
loss.backward()

# Store autograd gradients for comparison
autograd_weight_grad = conv_layer.weight.grad.clone()
autograd_bias_grad = conv_layer.bias.grad.clone()


# Manually compute the gradient with respect to W
with torch.no_grad():
    dL_dy_hat = 2 * (y_hat - y).sum()

    manual_dW = dL_dy_hat * x
    manual_db = dL_dy_hat

# Compare the gradients from autograd with manually computed ones
print("Autograd gradient for W:\n", autograd_weight_grad)
print("Manual gradient for W:\n", manual_dW)

print("\nAutograd gradient for b:\n", autograd_bias_grad)
print("Manual gradient for b:\n", manual_db)

# Check if they match
print("\nDifference in gradients for W: ", torch.abs(autograd_weight_grad - manual_dW).max().item())
print("Difference in gradients for b: ", torch.abs(autograd_bias_grad - manual_db).item())
