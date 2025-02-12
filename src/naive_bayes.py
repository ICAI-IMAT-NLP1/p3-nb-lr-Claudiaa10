import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.size(1) # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # Count number of samples for each output class and divide by total of samples
        class_priors: Dict[int, torch.Tensor] = {}
        class_counts = Counter(labels.tolist())
        total_samples = labels.shape[0]
        for class_label, class_count in class_counts.items():
            class_priors[class_label] = torch.tensor(class_count/total_samples)
        return class_priors
     

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        #Estimate conditional probabilities for the words in features and apply smoothing
        class_word_counts: Dict[int, torch.Tensor] = {}
        
        for class_label in torch.unique(labels):
            class_samples = features[labels == class_label]
            word_counts = class_samples.sum(dim = 0)
            total_words = word_counts.sum()
            smoothed_probs = (word_counts+delta)/(total_words+delta*self.vocab_size)
            class_word_counts[int(class_label)] = smoothed_probs

        return class_word_counts
        
      

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # Calculate posterior based on priors and conditional probabilities of the words
        log_posteriors: torch.Tensor = None
        log_posteriors_dict = {}
        list_log_post = []
        for class_label, class_prior in self.class_priors.items():
            log_conditional_probs = torch.log(self.conditional_probabilities[class_label])
            log_likelihood = torch.sum(feature * log_conditional_probs)
            log_posteriors_dict[class_label] = torch.log(class_prior) + log_likelihood

        for c in log_posteriors_dict.keys():
            list_log_post.append(log_posteriors_dict[c])
        
        log_posteriors: torch.Tensor = torch.tensor(list_log_post)
        return log_posteriors
            

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # Calculate log posteriors and obtain the class of maximum likelihood 
        log_posteriors = self.estimate_class_posteriors(feature)
        pred:int = torch.argmax(log_posteriors).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors = self.estimate_class_posteriors(feature)
        probs: torch.Tensor = torch.softmax(log_posteriors,dim=0)
        return probs
