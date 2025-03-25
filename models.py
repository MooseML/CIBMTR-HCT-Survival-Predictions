import torch
import torch.nn as nn
from config import CFG

class CoxResNet(nn.Module):
    def __init__(self, input_size, hidden_size=4096, dropout=0.05):
        super(CoxResNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.residual_block = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU(),
                                            nn.Dropout(dropout),
                                            nn.Linear(hidden_size, hidden_size),
                                            nn.BatchNorm1d(hidden_size),
                                            nn.ReLU())
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        residual = x.detach() # .detach() to avoid modifying computation graph
        x = self.residual_block(x)
        x = x + residual # non-in-place addition
        return self.output_layer(x)


class CoxPHLoss(nn.Module):
    def forward(self, risks, times, events):
        """Calculate the negative partial log-likelihood for Cox Proportional Hazard Loss"""
        order = torch.argsort(times, descending=True)
        sorted_risks = risks[order]
        sorted_events = events[order]

        hazards = torch.exp(sorted_risks) # exponential transformation to avoid large negative risks
        cum_hazards = torch.cumsum(hazards, dim=0)

        # don't take log(0) by adding small constant (numerical stability)
        log_cum_hazards = torch.log(cum_hazards + 1e-7)

        # calc partial likelihood
        partial_ll = sorted_events * (sorted_risks - log_cum_hazards)
        loss = -torch.mean(partial_ll)

        return loss


def train_cox_resnet_fullbatch(model, features, log_times, events, optimizer, criterion, epochs=CFG.cox_resnet_params['epochs'], device='cuda', sample_weights=None, clip_value=0.5):
    model.to(device)
    features = features.to(device)
    log_times = log_times.to(device)
    events = events.to(device)

    if sample_weights is not None:
        sample_weights = sample_weights.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(features).squeeze()

        # loss (apply sample weighting if provided)
        loss = criterion(preds, log_times, events)

        if sample_weights is not None:
            loss = (loss * sample_weights).mean() # sample weighting

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

