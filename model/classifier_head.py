import torch
import torch.nn as nn
from transformers.activations import get_activation


class SequenceClassifier(nn.Module):
    """
    A modular head for sentence-level classification tasks, designed to be attached to transformer-based models.

    Args:
        hidden_size (int): The size of the input features, typically the last hidden layer size of the transformer model.
        activation_function (str or callable): The activation function applied after the dense layer, e.g., 'tanh' or 'gelu'.
        num_labels (int): The number of classification labels.
        classifier_dropout (float, optional): Dropout probability for the classifier. Defaults to `dropout_prob` if not provided.
        dropout_prob (float): The dropout probability for the dropout layers.
    """

    def __init__(self, hidden_size, activation_function, num_labels, openmax=True, classifier_dropout=None,
                 dropout_prob=0.1):
        super(SequenceClassifier, self).__init__()
        self.openmax = openmax
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        self.activate = get_activation(activation_function)  # Ensuring activation function is correctly instantiated
        if self.openmax:

            self.activation_dense = nn.Linear(hidden_size, num_labels)
            self.openmax_out_proj = nn.Linear(num_labels, num_labels)
        print("Openmax is set to", self.openmax)

    def forward(self, features, **kwargs):
        """
        Forward pass for sentence classification.

        Args:
            features (torch.Tensor): Input features of shape `(batch_size, hidden_size)`.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output of the dense layer, of shape `(batch_size, hidden_size)`.
                - torch.Tensor: Logits for each class, of shape `(batch_size, num_labels)`.
        """

        if self.openmax:
            x = self.dense(features)
            x = self.dropout(x)
            x = self.activate(x)
            logits = self.out_proj(x)
            dense_output = logits.detach()  # 倒数第二层的激活向量
            return logits, dense_output
        else:
            x = self.dense(features)
            x = self.dropout(x)
            x = self.activate(x)
            logits = self.out_proj(x)
            return logits
