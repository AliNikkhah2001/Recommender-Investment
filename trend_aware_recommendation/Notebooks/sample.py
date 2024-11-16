## Model from ChatGPT: https://chatgpt.com/share/67390088-42a0-8009-bc56-823495c03fc6

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, GCNConv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

# Step 1: Load and Preprocess the Dataset
def load_data(file_path):
    # Load data with low_memory=False to avoid DtypeWarning and to handle mixed types properly
    data = pd.read_csv(file_path, low_memory=False)
    # Basic preprocessing: Handle missing values, normalize timestamps, etc.
    data.fillna(0, inplace=True)
    return data

# Load dataset
dataset = load_data('../data/crunchbase/investments_VC.csv')

# Step 2: Create Graph Structure
def create_graph(data):
    # Standardize column names: convert to lowercase and strip any extra whitespace
    data.columns = data.columns.str.strip().str.lower()

    # Since there are no clear 'organization' and 'brand' columns, we'll use 'name' and 'market'
    if 'name' not in data.columns or 'market' not in data.columns:
        raise ValueError("Expected columns 'name' and 'market' not found in the dataset.")

    # Map startups to unique IDs
    name_ids = {name: idx for idx, name in enumerate(data['name'].unique())}
    market_ids = {market: idx + len(name_ids) for idx, market in enumerate(data['market'].unique())}

    # Create edges based on the market relationships
    edges = []
    edge_types = []
    for _, row in data.iterrows():
        name = name_ids[row['name']]
        market = market_ids[row['market']]
        edges.append((name, market))
        edge_types.append(0)  # Edge type 0: operates in the same market

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return name_ids, market_ids, edge_index, edge_type

# Create graph structure
name_ids, market_ids, edge_index, edge_type = create_graph(dataset)

# Step 3: Define RGCN Layer for Preference Propagation
class RGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, num_bases=None):
        super(RGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations, num_bases)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations, num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        return x

# Step 4: Define GRU for Trend Extraction
class TrendExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TrendExtractor, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, hidden = self.gru(x)
        return hidden.squeeze(0)

# Step 5: Define the Overall Model
class ITRS(nn.Module):
    def __init__(self, num_entities, embedding_dim, num_relations, hidden_dim):
        super(ITRS, self).__init__()
        self.embedding = nn.Embedding(num_entities, embedding_dim)
        self.rgcn = RGCN(embedding_dim, hidden_dim, num_relations)
        self.trend_extractor = TrendExtractor(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_type, trends):
        x = self.embedding(x)
        x = self.rgcn(x, edge_index, edge_type)
        trends = self.trend_extractor(trends)
        combined = torch.cat([x, trends], dim=-1)
        score = self.mlp(combined)
        return score

# Step 6: Train and Evaluate the Model
def train_model(model, data, edge_index, edge_type, optimizer, criterion, epochs=50):
    model.train()
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        predictions = model(data.x, edge_index, edge_type, data.trends)
        loss = criterion(predictions, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Initialize Model
num_entities = len(name_ids) + len(market_ids)
embedding_dim = 40
hidden_dim = 40
num_relations = 1

model = ITRS(num_entities, embedding_dim, num_relations, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Prepare Data for Training
# Number of nodes (entities)
num_nodes = len(name_ids) + len(market_ids)

# Node indices (used for embedding lookup)
node_indices = torch.arange(num_nodes, dtype=torch.long)  # Use LongTensor for embedding

# Random binary labels (0 or 1) for the target variable
data_y = torch.randint(0, 2, (num_nodes, 1), dtype=torch.float)

# Random trends (assuming 10 time steps)
data_trends = torch.randn((num_nodes, 10, embedding_dim))

# Create a Data object to hold the features and labels
data = Data(x=node_indices, y=data_y, trends=data_trends)

# Train the model
train_model(model, data, edge_index, edge_type, optimizer, criterion)
