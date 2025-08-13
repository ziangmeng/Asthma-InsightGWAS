# train_transformer_asthma.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import argparse
import os
import random
from pathlib import Path
import pickle

# -------------------- Paths (relative to this script) --------------------
BASE_DIR = Path(__file__).resolve().parent  
DEFAULT_DATA_PATH = BASE_DIR / "data" / "Train_Data_Asthma.txt"
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "asthma_transformer.pth"

# -------------------- Argument Parser --------------------
parser = argparse.ArgumentParser(description='Train Transformer on Asthma-like dataset')

parser.add_argument('--data_path', type=str,
                    default=str(DEFAULT_DATA_PATH),
                    help='Path to training data (.txt, tab-separated). Default: ./data/Train_Data_Asthma.txt')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer layers')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of feedforward layer')
parser.add_argument('--d_model', type=int, default=20, help='Transformer model dimension (after input mapping)')
parser.add_argument('--threshold', type=float, default=0.9, help='Classification threshold for positive class')
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
parser.add_argument('--output_path', type=str,
                    default=str(DEFAULT_MODEL_PATH),
                    help='Path to save model. Default: ./model/asthma_transformer.pth')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

args = parser.parse_args()

# -------------------- Reproducibility --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(args.seed)

# -------------------- Load and Preprocess Data --------------------
df = pd.read_csv(args.data_path, sep='\t')

feature_cols = [
    'chr_hg38', 'bpos_GCST010042', 'ld', 'beta_GCST010042',
    'se_GCST010042', 'pval_GCST010042', 'n_GCST010042',
    'GTEx_V8_cis_eqtl', 'not_GTEx_eqtl', 'mqtl_EAS', 'mqtl_EUR',
    'sqtl', 'scqtl', 'consensus_footprints_and_motifs_hg38',
    'encode', 'report'
]
assert all(col in df.columns for col in feature_cols), "Some feature columns are missing!"
assert 'label' in df.columns, "Column 'label' not found!"

X = df[feature_cols]
y = df['label'].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

SCALER_PATH = Path(__file__).resolve().parent / "model" / "scaler.pkl"
SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {SCALER_PATH}")


X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)

val_size = int(args.val_split * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(args.seed)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# -------------------- Define Model --------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, d_model=20, output_dim=1):
        super().__init__()
        self.input_mapping = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_mapping(x)          # (batch, d_model)
        x = x.unsqueeze(1)                 # (batch, seq_len=1, d_model)
        x = x.permute(1, 0, 2)             # (seq_len=1, batch, d_model)
        x = self.transformer(x)            # (1, batch, d_model)
        x = x[0, :, :]                     # (batch, d_model)
        x = self.fc(x)                     # (batch, 1)
        x = self.sigmoid(x)
        return x

input_dim = X_tensor.shape[1]
model = TransformerClassifier(input_dim=input_dim,
                              num_heads=args.num_heads,
                              num_layers=args.num_layers,
                              hidden_dim=args.hidden_dim,
                              d_model=args.d_model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# -------------------- Training Loop --------------------
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)              
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()

            predicted = (outputs > args.threshold).float()
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    train_loss = running_loss / max(1, len(train_loader))
    val_loss_avg = val_loss / max(1, len(val_loader))
    acc = 100.0 * correct / max(1, total)

    print(f"Epoch {epoch+1}/{args.num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | "
          f"Val Acc(@{args.threshold:.2f}): {acc:.2f}%")

# -------------------- Save Model --------------------
out_path = Path(args.output_path)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), str(out_path))
print(f"Model saved to {out_path}")
