import torch.nn as nn

# CUstom model which adds a classification head
class FrozenBertClassifier(nn.Module):
    def __init__(self, base_model, hidden_size=768, num_labels=5, pos_weight = None):
        super().__init__()
        # give the model the pretrained model with no head
        self.base_model = base_model

        # Add dropout to reduce overfitting
        self.dropout = nn.Dropout(0.2)
        # Custom classification head which provides final outputs
        self.classifier = nn.Sequential(
            # reduce the size of the input to the head
            nn.Linear(hidden_size, 128),
            # Help model learn patterns and not just straight lines
            nn.ReLU(),  
            # normalization
            nn.LayerNorm(128),
            # Another layer of dropout to reduce overfitting
            nn.Dropout(0.3),
            # This outputs the final logits for each label
            nn.Linear(128, num_labels)
        )

        # Loss function for multi label classification
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        # Provide the inputs to the basemodel
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # get the output from the basemodel
        cls_output = getattr(outputs, "pooler_output", outputs.last_hidden_state[:, 0, :])  # [CLS] token
        # use the dropout layer to prevent Overfitting
        x = self.dropout(cls_output)
        # feed the output to the head and get the output from the head
        logits = self.classifier(x)
        # calculate loss 
        loss = self.loss_fn(logits, labels.float())
        
        # retun the loss and the output logits
        return {'loss': loss, 'logits': logits} if loss is not None else logits
