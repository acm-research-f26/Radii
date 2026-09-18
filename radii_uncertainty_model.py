import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CalibratedMedicalRegressor(nn.Module):
    """
    Dual-head regression network inspired by Laves et al. (2020)
    Modified Architecture:
    - Backbone: ResNet18 (swappable feature extractor)
    - MC Dropout Layer for epistemic uncertainty estimation
    - Decoupled Heads: Separate MLP branches for Mean (y_hat) and Log-Variance (log_sigma^2)
    """
    def __init__(self, backbone="resnet18", dropout_rate=0.2):
        super().__init__()
        
        # 1. Feature Extractor Backbone
        if backbone == "resnet18":
            base_model = models.resnet18(weights=None)
            feature_dim = base_model.fc.in_features
            # Remove original classification head
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # 2. Epistemic Uncertainty via Dropout (Active at test time)
        self.dropout = nn.Dropout(p=dropout_rate)

        # 3. Architectural Modification: Decoupled Non-Linear Regression Heads
        # Head A: Target Prediction (e.g., Fibroid / Organ Volume or Diameter)
        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Head B: Aleatoric Uncertainty (log(sigma^2))
        self.variance_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Extract visual features: [Batch, feature_dim, 1, 1] -> [Batch, feature_dim]
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        features = self.dropout(features)

        # Compute mean and log-variance
        y_hat = self.mean_head(features)
        log_var = self.variance_head(features)
        
        return y_hat.squeeze(-1), log_var.squeeze(-1)


def gaussian_nll_loss(y_true, y_pred, log_var):
    """
    Gaussian Negative Log-Likelihood Loss from Laves et al.
    Trains network to output accurate mean and aleatoric uncertainty simultaneously.
    """
    precision = torch.exp(-log_var)
    loss = 0.5 * precision * (y_true - y_pred) ** 2 + 0.5 * log_var
    return torch.mean(loss)


if __name__ == "__main__":
    print("Testing Modified Calibrated Medical Regressor...")
    model = CalibratedMedicalRegressor(dropout_rate=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulate batch of 8 MRI slice images: (batch_size, channels, H, W)
    dummy_mri_batch = torch.randn(8, 3, 224, 224)
    # Target volume or diameter (e.g., cm^3)
    dummy_targets = torch.tensor([15.2, 42.0, 31.5, 8.4, 60.1, 22.8, 18.0, 45.2])

    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred, log_var = model(dummy_mri_batch)
    sigma = torch.sqrt(torch.exp(log_var))
    
    # Loss computation
    loss = gaussian_nll_loss(dummy_targets, y_pred, log_var)
    loss.backward()
    optimizer.step()

    print(f"Training Step Successful! Batch Loss: {loss.item():.4f}")
    print(f"Sample Predicted Values: {y_pred.detach().numpy()[:3]}")
    print(f"Sample Predicted Sigma:  {sigma.detach().numpy()[:3]}")