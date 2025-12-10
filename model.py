"""
MentalRoBERTa-Caps: A Capsule-Enhanced Transformer Model for Mental Health Classification
Based on: Wagay et al. (2025) - https://pmc.ncbi.nlm.nih.gov/articles/PMC12284574/

This implementation recreates the architecture described in the paper:
- 6-layer MentalRoBERTa encoder (from mental/mental-roberta-base)
- Capsule Network layer with dynamic routing
- Classification head for mental health conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer


class CapsuleLayer(nn.Module):
    """
    Capsule Layer with Dynamic Routing
    
    As described in the paper:
    - Primary capsules are generated from the [CLS] token output
    - Dynamic routing aggregates votes from primary to class capsules
    - The length of each class capsule represents class probability
    """
    
    def __init__(self, input_dim=768, num_primary_caps=8, primary_cap_dim=16, 
                 num_class_caps=5, class_cap_dim=16, num_routing_iterations=3):
        super(CapsuleLayer, self).__init__()
        
        self.num_primary_caps = num_primary_caps
        self.primary_cap_dim = primary_cap_dim
        self.num_class_caps = num_class_caps
        self.class_cap_dim = class_cap_dim
        self.num_routing_iterations = num_routing_iterations
        
        # Primary capsule projections (Equation 11 in paper)
        self.primary_caps = nn.ModuleList([
            nn.Linear(input_dim, primary_cap_dim) for _ in range(num_primary_caps)
        ])
        
        # Weight matrices for vote generation (Equation 13)
        # W_ij transforms capsule i's output to predict capsule j
        self.W = nn.Parameter(torch.randn(num_primary_caps, num_class_caps, 
                                          primary_cap_dim, class_cap_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W.view(num_primary_caps * num_class_caps, 
                                            primary_cap_dim, class_cap_dim).view(-1, class_cap_dim))
    
    def squash(self, s, dim=-1):
        """
        Squash activation function (Equation 12 in paper)
        Ensures vector magnitude is between 0 and 1 while preserving direction
        """
        squared_norm = (s ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * s / (torch.sqrt(squared_norm) + 1e-8)
    
    def forward(self, x):
        """
        Forward pass through capsule layer
        
        Args:
            x: [batch_size, input_dim] - CLS token representation from RoBERTa
        
        Returns:
            class_caps: [batch_size, num_class_caps, class_cap_dim] - Class capsule outputs
        """
        batch_size = x.size(0)
        
        # Generate primary capsules (Equation 11)
        primary_outputs = []
        for caps_layer in self.primary_caps:
            u = self.squash(caps_layer(x))  # [batch_size, primary_cap_dim]
            primary_outputs.append(u)
        
        # Stack primary capsules: [batch_size, num_primary_caps, primary_cap_dim]
        u = torch.stack(primary_outputs, dim=1)
        
        # Compute vote vectors (Equation 13)
        # u: [batch_size, num_primary_caps, 1, 1, primary_cap_dim]
        u_expand = u.unsqueeze(2).unsqueeze(3)
        
        # W: [1, num_primary_caps, num_class_caps, primary_cap_dim, class_cap_dim]
        W_expand = self.W.unsqueeze(0)
        
        # u_hat: [batch_size, num_primary_caps, num_class_caps, class_cap_dim]
        u_hat = torch.matmul(u_expand, W_expand).squeeze(3)
        
        # Dynamic routing (Equations 14-15)
        # Initialize routing logits b_ij to zero
        b = torch.zeros(batch_size, self.num_primary_caps, self.num_class_caps, 
                        device=x.device)
        
        for iteration in range(self.num_routing_iterations):
            # Coupling coefficients c_ij (Equation 14)
            c = F.softmax(b, dim=2)  # [batch_size, num_primary_caps, num_class_caps]
            
            # Weighted sum of votes (Equation 15)
            # c: [batch_size, num_primary_caps, num_class_caps, 1]
            # u_hat: [batch_size, num_primary_caps, num_class_caps, class_cap_dim]
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)  # [batch_size, num_class_caps, class_cap_dim]
            
            # Squash to get class capsule outputs
            v = self.squash(s, dim=-1)  # [batch_size, num_class_caps, class_cap_dim]
            
            # Update routing logits (except on last iteration)
            if iteration < self.num_routing_iterations - 1:
                # Agreement: dot product between prediction and output
                # v: [batch_size, 1, num_class_caps, class_cap_dim]
                # u_hat: [batch_size, num_primary_caps, num_class_caps, class_cap_dim]
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement
        
        return v


class MentalRoBERTaCaps(nn.Module):
    """
    MentalRoBERTa-Caps Model
    
    Architecture (as per paper):
    - BERT/RoBERTa encoder (using only 6 of 12 layers)
    - Capsule layer for hierarchical feature learning
    - Linear classifier for final prediction
    
    Supported models:
    - German: "deepset/gbert-base", "bert-base-german-cased", "uklfr/gottbert-base"
    - English: "roberta-base", "mental/mental-roberta-base" (requires access)
    - Multilingual: "xlm-roberta-base"
    
    ~125.1 million trainable parameters, ~1.15 GB memory footprint
    """
    
    def __init__(self, 
                 num_classes=5,
                 num_layers=6,
                 num_primary_caps=8,
                 primary_cap_dim=16,
                 class_cap_dim=16,
                 num_routing_iterations=3,
                 model_name="deepset/gbert-base"):  # Default: German BERT
        super(MentalRoBERTaCaps, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Load MentalRoBERTa
        self.roberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # We'll use only the first num_layers encoder layers
        # This is done in forward by extracting intermediate layers
        
        # Capsule layer
        self.capsule = CapsuleLayer(
            input_dim=self.hidden_size,
            num_primary_caps=num_primary_caps,
            primary_cap_dim=primary_cap_dim,
            num_class_caps=num_classes,
            class_cap_dim=class_cap_dim,
            num_routing_iterations=num_routing_iterations
        )
        
        # Final classifier (Equation 17)
        self.classifier = nn.Linear(num_classes * class_cap_dim, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_classes]
            capsule_outputs: [batch_size, num_classes, class_cap_dim]
        """
        # Get RoBERTa outputs with hidden states
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use output from layer num_layers (0-indexed, +1 for embeddings)
        # hidden_states[0] = embeddings, hidden_states[1] = layer 1, etc.
        hidden_states = outputs.hidden_states
        
        # Get the output of the specified layer (using 6 layers = index 6)
        layer_output = hidden_states[min(self.num_layers, len(hidden_states) - 1)]
        
        # Extract [CLS] token representation
        cls_output = layer_output[:, 0, :]  # [batch_size, hidden_size]
        cls_output = self.dropout(cls_output)
        
        # Pass through capsule layer
        capsule_outputs = self.capsule(cls_output)  # [batch_size, num_classes, class_cap_dim]
        
        # Flatten capsule outputs for classification
        caps_flat = capsule_outputs.view(capsule_outputs.size(0), -1)
        
        # Final classification
        logits = self.classifier(caps_flat)
        
        return logits, capsule_outputs
    
    def get_capsule_lengths(self, capsule_outputs):
        """
        Get the length of each class capsule (represents class probability)
        """
        return torch.sqrt((capsule_outputs ** 2).sum(dim=-1))


class MentalRoBERTaCapsClassifier:
    """
    High-level classifier wrapper for easy inference
    """
    
    # Class labels for different datasets
    SWMH_LABELS = ['depression', 'anxiety', 'bipolar', 'SuicideWatch', 'offmychest']
    DREADDIT_LABELS = ['not_stressed', 'stressed']
    SAD_LABELS = ['school', 'financial_problem', 'family_issues', 'social_relationships',
                  'work', 'health_issues', 'emotional_turmoil', 'everyday_decision_making', 'other']
    
    def __init__(self, num_classes=5, labels=None, device=None, model_name="deepset/gbert-base"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.labels = labels or self.SWMH_LABELS[:num_classes]
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = MentalRoBERTaCaps(num_classes=num_classes, model_name=model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, text):
        """
        Preprocess text as described in paper:
        - Remove URLs, mentions, special characters
        - Tokenize with WordPiece tokenizer
        """
        import re
        
        # Text cleaning (as per paper's preprocessing pipeline)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        
        return text
    
    def predict(self, text, return_probs=True):
        """
        Predict mental health category for input text
        
        Args:
            text: Input text string
            return_probs: Whether to return probability distribution
        
        Returns:
            prediction: Predicted label
            probabilities: (optional) Probability for each class
        """
        # Preprocess
        clean_text = self.preprocess(text)
        
        # Tokenize
        inputs = self.tokenizer(
            clean_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            logits, capsule_outputs = self.model(input_ids, attention_mask)
            
            # Get probabilities using capsule lengths
            caps_lengths = self.model.get_capsule_lengths(capsule_outputs)
            probs = F.softmax(logits, dim=-1)
            
            # Get prediction
            pred_idx = torch.argmax(probs, dim=-1).item()
            prediction = self.labels[pred_idx]
        
        if return_probs:
            prob_dict = {label: probs[0, i].item() for i, label in enumerate(self.labels)}
            return prediction, prob_dict
        
        return prediction
    
    def batch_predict(self, texts):
        """
        Predict for multiple texts at once
        """
        results = []
        for text in texts:
            pred, probs = self.predict(text)
            results.append({'text': text, 'prediction': pred, 'probabilities': probs})
        return results


if __name__ == "__main__":
    # Quick test
    print("Testing MentalRoBERTa-Caps model architecture...")
    
    # Create model
    model = MentalRoBERTaCaps(num_classes=5, num_layers=6)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Estimated memory: ~{total_params * 4 / 1e9:.2f} GB (float32)")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    dummy_input = torch.randint(0, 50265, (batch_size, seq_len))
    dummy_mask = torch.ones(batch_size, seq_len)
    
    print(f"\nTest forward pass with input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        logits, caps = model(dummy_input, dummy_mask)
    
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Capsule outputs shape: {caps.shape}")
    print("\n✓ Model architecture test passed!")
