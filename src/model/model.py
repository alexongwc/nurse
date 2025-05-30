# model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SimpleGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x

def train_gat(
    data, 
    in_dim, 
    hidden_dim, 
    out_dim, 
    heads, 
    epochs, 
    lr,
    verbose=True
):
    """
    Train GAT model on given data object.
    Hyperparameters must be provided as arguments.
    """
    model = SimpleGAT(in_dim, hidden_dim, out_dim, heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        edge_emb = (out[data.edge_index[0]] + out[data.edge_index[1]]) / 2
        pred = edge_emb.squeeze()
        loss = F.binary_cross_entropy_with_logits(pred, data.y)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
    return model

@torch.no_grad()
def predict_gat(model, data):
    """
    Returns probabilities for each edge in the data object.
    """
    model.eval()
    out = model(data)
    edge_emb = (out[data.edge_index[0]] + out[data.edge_index[1]]) / 2
    logits = edge_emb.squeeze()
    probs = torch.sigmoid(logits)
    return probs.cpu().numpy()