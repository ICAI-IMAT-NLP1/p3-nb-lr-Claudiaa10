import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ):
        """
        Train the logistic regression model using pre-processed features and labels.

        Args:
            features (torch.Tensor): The bag of words representations of the training examples.
            labels (torch.Tensor): The target labels.
            learning_rate (float): The learning rate for gradient descent.
            epochs (int): The number of iterations over the training dataset.

        Returns:
            None: The function updates the model weights in place.
        """
        # Implement gradient-descent algorithm to optimize logistic regression weights
        features = torch.cat((features, torch.ones(features.shape[0], 1)), dim=1)

        if self._weights is None:
            self._weights = self.initialize_parameters(features.shape[1] - 1, self.random_state)

        for epoch in range(epochs):
            linear_combination = torch.matmul(features, self._weights)
            predictions = self.sigmoid(linear_combination)
            loss = self.binary_cross_entropy_loss(predictions, labels)
            gradient = torch.matmul(features.T, (predictions - labels)) / labels.shape[0]
            self._weights -= learning_rate * gradient

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        """
        Predict class labels for given examples based on a cutoff threshold.

        Args:
            features (torch.Tensor): The bag of words representations of the examples to classify.
            cutoff (float): The threshold for classifying a sample as class 1.

        Returns:
            torch.Tensor: Predicted class labels (0 or 1).
        """
        probabilities: torch.Tensor = self.predict_proba(features)
        decisions: torch.Tensor = torch.where(probabilities > cutoff, 1.0, 0.0)
        return decisions

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution of class labels for given examples.

        Args:
            features (torch.Tensor): The bag of words representations of the examples to classify.

        Returns:
            torch.Tensor: Predicted probabilities for class 1 (in the binary classification case).
        
        Raises:
            ValueError: If the model weights are not initialized (model not trained).
        """
        if self._weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")
        features = torch.cat((features, torch.ones(features.shape[0], 1)), dim=1)
        z = torch.matmul(features, self._weights)
        probabilities: torch.Tensor = self.sigmoid(z)

        return probabilities

    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        """
        Initialize the weights of the logistic regression model.

        Args:
            dim (int): The number of features (input dimensions).
            random_state (int): The seed for random number generation.

        Returns:
            torch.Tensor: Initialized weights as a tensor with size (dim + 1,).
        """
        torch.manual_seed(random_state)
        params: torch.Tensor = torch.randn(dim + 1, dtype=torch.float32)
        return params

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """
        Compute the sigmoid of z.

        Args:
            z (torch.Tensor): The input tensor to the sigmoid function.

        Returns:
            torch.Tensor: The sigmoid of z.
        """
        result: torch.Tensor = (1 / (1 + torch.exp(-z)))
        return result

    @staticmethod
    def binary_cross_entropy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the binary cross-entropy loss for the model's predictions.

        Args:
            predictions (torch.Tensor): The predicted probabilities for class 1.
            targets (torch.Tensor): The true labels (0 or 1).

        Returns:
            torch.Tensor: The computed binary cross-entropy loss.
        """
        ce_loss: torch.Tensor = -torch.mean(targets * torch.log(predictions + 1e-17) + (1 - targets) * torch.log(1 - predictions + 1e-17))
        return ce_loss

    @property
    def weights(self):
        """Get the weights of the logistic regression model."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Set the weights of the logistic regression model."""
        self._weights: torch.Tensor = value
