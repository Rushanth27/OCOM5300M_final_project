import torch
import json
from typing import Dict, List

import torch
import json
from typing import Dict, List
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import joblib

class RiskGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(RiskGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GcnRiskRecognition:
    def __init__(self, llm_client, model_path: str = 'scope_coordinator/models/saved_models'):
        self.client = llm_client
        self.model = RiskGCN(num_features=1, num_classes=4)
        self.label_encoder = None
        try:
            self.model.load_state_dict(torch.load(f'{model_path}/risk_gcn_model.pth'))
            self.label_encoder = joblib.load(f'{model_path}/label_encoder.joblib')
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load GCN model: {str(e)}")

    def predict_risk(self, discussion: List[Dict]) -> Dict:
        """
        Predicts risk level based on the discussion content
        """
        if not discussion:
            return {"error": "Empty discussion"}

        try:
            # Extract text content from discussion
            project_scope = " ".join([
                msg["content"][0]["text"]
                for msg in discussion
                if "content" in msg and len(msg["content"]) > 0
            ])

            # Create input tensor
            x = torch.tensor([[len(project_scope)]], dtype=torch.float)
            edge_index = torch.tensor([[0, 0]], dtype=torch.long).t()
            data = Data(x=x, edge_index=edge_index)

            # Make prediction
            with torch.no_grad():
                output = self.model(data)
                probabilities = torch.exp(output)
                prediction = output.argmax(dim=1)

                risk_level = self.label_encoder.inverse_transform(prediction.numpy())[0]

                prob_dict = {}
                for i, prob in enumerate(probabilities[0]):
                    risk_class = self.label_encoder.inverse_transform([i])[0]
                    prob_dict[risk_class] = float(prob)

                return {
                    'predicted_risk_level': risk_level,
                    'probabilities': prob_dict
                }

        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}

    def _get_default_args(self):
        class Args:
            def __init__(self):
                self.wp = 10
                self.wf = 10
                self.device = torch.device('cpu')
                self.dropout = 0.1
                self.n_speakers = 2

        return Args()

    def _preprocess_discussion(self, discussion: List[Dict]) -> Dict:
        """
        Converts discussion to tensor format needed by GCN.
        """
        # Ensure we have at least one item
        if not discussion:
            discussion = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Empty discussion"
                }]
            }]

        batch_size = 1
        seq_len = len(discussion)
        input_dim = 100  # Match your model's input dimension

        # Initialize tensors with proper dimensions
        text_tensor = torch.zeros((batch_size, seq_len, input_dim))
        text_len_tensor = torch.tensor([seq_len], dtype=torch.long)
        speaker_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)
        label_tensor = torch.zeros((batch_size, seq_len), dtype=torch.long)
        adj_matrix = torch.zeros((batch_size, seq_len, seq_len))  # Add adjacency matrix

        # Fill in the tensors
        for i, entry in enumerate(discussion):
            # Extract text from the nested structure
            text = ""
            if isinstance(entry, dict):
                content = entry.get("content", [])
                if content and isinstance(content, list):
                    first_content = content[0]
                    if isinstance(first_content, dict):
                        text = first_content.get("text", "")

            # Create a simple embedding based on the text
            # For now, using random embeddings - in production, use a proper embedding model
            text_embedding = torch.randn(input_dim)  # Random embedding for now
            text_tensor[0, i] = text_embedding

            # Set speaker IDs (0 for user, 1 for assistant)
            speaker_id = 1 if entry.get("role") == "assistant" else 0
            speaker_tensor[0, i] = speaker_id

            # Create connections in adjacency matrix
            # Connect each utterance to previous and next utterance
            if i > 0:
                adj_matrix[0, i, i - 1] = 1
                adj_matrix[0, i - 1, i] = 1

        return {
            "text_tensor": text_tensor,
            "text_len_tensor": text_len_tensor,
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "adj_matrix": adj_matrix,
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.bool)  # Add attention mask
        }

    def _format_gcn_predictions(self, predictions: torch.Tensor, mask: torch.Tensor) -> List[Dict]:
        """
        Converts GCN predictions to readable format
        """
        risk_categories = {
            0: "NO_RISK",
            1: "LOW_RISK",
            2: "MODERATE_RISK",
            3: "HIGH_RISK",
            4: "SEVERE_RISK",
            5: "CRITICAL_RISK"
        }

        results = []

        # Ensure predictions and mask are not empty
        if predictions.numel() > 0 and mask.numel() > 0:
            valid_predictions = predictions[mask.bool()]

            for i, pred in enumerate(valid_predictions):
                results.append({
                    "risk_level": risk_categories.get(pred.item(), "UNKNOWN_RISK"),
                    "index": i
                })

        return results

    def identify_risks_in_project_using_GCN(self, project_discussion: List[Dict]) -> List[Dict]:
        try:
            # Convert discussion to format needed by GCN
            processed_data = self._preprocess_discussion(project_discussion)

            # Get GCN risk analysis
            try:
                with torch.no_grad():
                    logits, mask = self.gcn_model(processed_data)
                    predictions = torch.argmax(logits, dim=-1)
                gcn_risks = self._format_gcn_predictions(predictions, mask)
            except Exception as e:
                print(f"Warning: GCN analysis failed: {str(e)}")
                gcn_risks = []

            # Get detailed risk analysis from Bedrock
            system_prompt = """Analyze the following project requirements and provide a detailed risk assessment. 
            Consider technical, operational, and business risks. Include:
            1. Technical risks (e.g., integration challenges, security concerns)
            2. Operational risks (e.g., deployment, maintenance)
            3. Business risks (e.g., user adoption, compliance)

            Format each risk as a JSON object with:
            - category: risk category (TECHNICAL/OPERATIONAL/BUSINESS)
            - severity: risk severity (LOW/MEDIUM/HIGH)
            - description: detailed description of the risk
            - mitigation: suggested mitigation strategy"""

            # Extract text from the discussion format
            project_description = ""
            if project_discussion:
                first_message = project_discussion[0]
                if isinstance(first_message, dict):
                    content = first_message.get("content", [])
                    if content and isinstance(content, list):
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            project_description = first_content.get("text", "")

            # Prepare context for Bedrock
            context = {
                "project_description": project_description,
                "initial_risks": gcn_risks
            }

            messages = [{"role": "user", "content": json.dumps(context)}]

            try:
                _, response = self.client.run(system_prompt, messages)
                detailed_risks = json.loads(response)
            except json.JSONDecodeError:
                # If response isn't JSON, format it as structured data
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                detailed_risks = []
                current_risk = {}

                for line in lines:
                    if line.startswith(('Technical:', 'Operational:', 'Business:')):
                        if current_risk:
                            detailed_risks.append(current_risk)
                        category = line.split(':')[0].upper()
                        current_risk = {
                            "category": category,
                            "severity": "MEDIUM",
                            "description": line.split(':', 1)[1].strip(),
                            "mitigation": "Needs assessment"
                        }
                    elif current_risk:
                        current_risk["description"] += " " + line

                if current_risk:
                    detailed_risks.append(current_risk)

            if not detailed_risks:
                detailed_risks = [{
                    "category": "TECHNICAL",
                    "severity": "MEDIUM",
                    "description": "Initial risk assessment needed",
                    "mitigation": "Conduct detailed technical review"
                }]

            return detailed_risks

        except Exception as e:
            print(f"Error in risk identification: {str(e)}")
            return [{
                "category": "ERROR",
                "severity": "HIGH",
                "description": f"Error in risk analysis: {str(e)}",
                "mitigation": "Review error and retry analysis"
            }]