import torch.nn as nn
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=8):
        super(MultiLabelClassifier, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, num_classes))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, concatenated_embeddings):
        # Flatten the input to (batch_size, -1) to match the input size of the linear layer
        
        flattened_input = concatenated_embeddings.view(concatenated_embeddings.size(0), -1)

        # Pass through the feed-forward network
        output = self.network(flattened_input)

        return output
