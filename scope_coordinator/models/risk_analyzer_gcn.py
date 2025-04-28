import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.data import Data

import os
from datetime import datetime

from transformers import AutoTokenizer, AutoModel

import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class RiskAnalyzerGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=6):  # Changed to 6 for scores 0-5
        super(RiskAnalyzerGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

        # Separate heads for impact and likelihood
        self.impact_head = nn.Linear(hidden_dim, num_classes)
        self.likelihood_head = nn.Linear(hidden_dim, num_classes)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Ensure edge_index is within bounds
        num_nodes = x.size(0)
        edge_index = edge_index.clamp(0, num_nodes - 1)

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        impact = F.log_softmax(self.impact_head(x), dim=1)
        likelihood = F.log_softmax(self.likelihood_head(x), dim=1)

        return impact, likelihood


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device=None):
        """
        Class method to load the model from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file
            device (torch.device, optional): Device to load the model to

        Returns:
            RiskAnalyzerGCN: Loaded model instance
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Get model parameters from checkpoint if they exist
            model_params = {}
            if 'model_config' in checkpoint:
                model_params = checkpoint['model_config']

            # Create a new instance of the model
            model = cls(
                input_dim=model_params.get('input_dim', 768),
                hidden_dim=model_params.get('hidden_dim', 64),
                num_classes=model_params.get('num_classes', 6)
            )

            # Load the state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # Move model to device and set to eval mode
            model = model.to(device)
            model.eval()

            return model

        except Exception as e:
            raise Exception(f"Error loading model from checkpoint: {str(e)}")


    def save_checkpoint(self, save_path):
        """
        Save the model to a checkpoint file.

        Args:
            save_path (str): Path where to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_classes': self.num_classes
            }
        }
        torch.save(checkpoint, save_path)

def load_text_embedding_model():
    """Load the BERT model for text embeddings"""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model


def get_bert_embedding(text, tokenizer, model):
    """Convert text to BERT embeddings"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def create_graph_data(df, tokenizer, model):
    """Create a graph from the dataset"""
    # Convert texts to embeddings
    embeddings = []
    total = len(df)
    last_percentage = 0

    for i, text in enumerate(df['Risk_Description']):
        embedding = get_bert_embedding(text, tokenizer, model)
        embeddings.append(embedding)

        # Calculate current percentage
        current_percentage = ((i + 1) / total) * 100

        # Print every 10%
        if int(current_percentage) // 10 > last_percentage // 10:
            print(f"Processing embeddings: {int(current_percentage)}% completed")
            last_percentage = current_percentage

    # Stack embeddings
    x = torch.stack(embeddings)

    # Create edges (connecting each node to itself)
    num_nodes = len(df)
    edge_index = []
    for i in range(num_nodes):
        edge_index.append([i, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    # Convert labels to tensors
    impact = torch.tensor(df['Impact'].values, dtype=torch.long)
    likelihood = torch.tensor(df['Likelihood'].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, impact=impact, likelihood=likelihood)


def load_and_process_data():
    """Load and prepare data from project_risks.csv"""
    # Load CSV file
    csv_path = 'scope_coordinator/models/project_risks.csv'
    df = pd.read_csv(csv_path)

    # Debug: Print column names
    print("Available columns:", df.columns.tolist())
    print("\nFirst few rows of data:")
    print(df.head())

    # Load BERT model
    tokenizer, model = load_text_embedding_model()

    # Create graph data
    data = create_graph_data(df, tokenizer, model)

    return data


def split_data(data, val_ratio=0.2):
    num_samples = len(data.likelihood)
    indices = torch.randperm(num_samples)

    # Calculate split point
    split_idx = int(num_samples * (1 - val_ratio))

    # Create train indices and validation indices
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create training data
    train_data = Data(
        x=data.x[train_indices],
        edge_index=data.edge_index,
        impact=data.impact[train_indices],
        likelihood=data.likelihood[train_indices]
    )

    # Create validation data
    val_data = Data(
        x=data.x[val_indices],
        edge_index=data.edge_index,
        impact=data.impact[val_indices],
        likelihood=data.likelihood[val_indices]
    )

    return train_data, val_data


def save_model(model, metrics=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "models/saved"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"risk_analyzer_gcn_{timestamp}.pt")

    save_dict = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics
    }

    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")


def evaluate_model(model, data):
    model.eval()
    impact_preds = []
    likelihood_preds = []
    impact_confidences = []
    likelihood_confidences = []

    with torch.no_grad():
        impact_out, likelihood_out = model(data)

        # Apply softmax to get probabilities
        impact_probs = torch.softmax(impact_out, dim=1)
        likelihood_probs = torch.softmax(likelihood_out, dim=1)

        # Get predictions (highest probability class)
        impact_pred = impact_probs.argmax(dim=1)
        likelihood_pred = likelihood_probs.argmax(dim=1)

        # Get confidence scores (maximum probability)
        impact_conf = impact_probs.max(dim=1)[0]
        likelihood_conf = likelihood_probs.max(dim=1)[0]

        # Convert to lists
        impact_preds = impact_pred.cpu().numpy()
        likelihood_preds = likelihood_pred.cpu().numpy()
        impact_confidences = impact_conf.cpu().numpy()
        likelihood_confidences = likelihood_conf.cpu().numpy()

    return impact_preds, likelihood_preds, impact_confidences, likelihood_confidences


def risk_classifier(risk_text, model_dir, temperature=1.0):
    """
    Predict impact and likelihood for a given risk text using the GCN model.
    
    Args:
        risk_text (str): Description of the risk to analyze
        temperature (float, optional): Temperature for softmax scaling. Defaults to 1.0.
    
    Returns:
        dict: Dictionary containing impact and likelihood predictions with confidence scores
    """
    # Define GCN model class
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, out_channels)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, training=self.training)
            return self.conv2(x, edge_index)
    
    # Load saved models
    vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    
    # Process input text
    risk_tfidf = vectorizer.transform([risk_text]).toarray()
    x = torch.tensor(risk_tfidf, dtype=torch.float)
    
    # Create a simple self-loop edge for single node prediction
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(x.shape[1], out_channels=len(label_encoder.classes_))
    model.load_state_dict(torch.load(os.path.join(model_dir, "gcn_model.pth"), 
                                     map_location=device))
    model.to(device)
    data = data.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(data)

        probabilities = F.softmax(output / temperature, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

        # print(probabilities)
        # print(prediction)
    
    # Create response in the same format as before
    result = {
            'value': prediction,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
    }
    
    return result

def main():
    # Hyperparameters
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    HIDDEN_DIM = 64
    INPUT_DIM = 768  # BERT base hidden size

    # Load and split data
    print("Loading data...")
    full_data = load_and_process_data()
    train_data, val_data = split_data(full_data, val_ratio=0.2)

    # Initialize model and optimizer
    model = RiskAnalyzerGCN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training model...")
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        impact_out, likelihood_out = model(train_data)

        impact_loss = F.nll_loss(impact_out, train_data.impact)
        likelihood_loss = F.nll_loss(likelihood_out, train_data.likelihood)
        total_loss = impact_loss + likelihood_loss

        total_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_impact_out, val_likelihood_out = model(val_data)
            val_impact_loss = F.nll_loss(val_impact_out, val_data.impact)
            val_likelihood_loss = F.nll_loss(val_likelihood_out, val_data.likelihood)
            val_loss = val_impact_loss + val_likelihood_loss

            if epoch % 10 == 0:
                print(f'Epoch {epoch}:')
                print(f'  Train Loss: {total_loss.item():.4f}')
                print(f'  Val Loss: {val_loss.item():.4f}')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_model(model, {'epoch': epoch, 'val_loss': val_loss.item()})
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after epoch {epoch}")
                    break

    print("\nPerforming final evaluation...")
    impact_preds, likelihood_preds, impact_conf, likelihood_conf = evaluate_model(model, val_data)

    # Load BERT model for predictions
    tokenizer, bert_model = load_text_embedding_model()

    print("\nTesting predictions with example risks...")
    test_risks = [
        "Critical system failure in production environment",
        "Minor documentation update required",
        "Potential security vulnerability in authentication system",
        "Network connectivity issues in non-critical system",
        "Data backup system failure"
    ]

    for risk in test_risks:
        result = predict_risk(risk, model, tokenizer, bert_model)
        print(f"\nRisk Description: {risk}")
        print(f"Predicted Impact: {result['impact']['value']} (Confidence: {result['impact']['confidence']:.2f})")
        print(
            f"Predicted Likelihood: {result['likelihood']['value']} (Confidence: {result['likelihood']['confidence']:.2f})")
        print(f"Risk Score: {result['risk_score']}")

if __name__ == "__main__":
    # main()
    impact_model_path = "scope_coordinator/models/saved_models/gcn_model_impact/"
    print(risk_classifier("This is a test risk example", impact_model_path))
    # print("Current working directory:", os.getcwd())
    # full_path = os.path.join(os.getcwd(), "scope_coordinator/models/saved_models/gcn_model/vectorizer.pkl")
    # print("Trying to access:", full_path)
    # print("File exists:", os.path.exists(full_path))
