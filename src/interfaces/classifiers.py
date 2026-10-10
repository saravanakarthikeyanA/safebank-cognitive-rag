from abc import ABC, abstractmethod

class BaseIntentClassifier(ABC):
    @abstractmethod
    def predict(self, query: str) -> dict:
        """
        Predict the intent category and sentiment of a user query.

        Args:
            query: The user query string.

        Returns:
            dict: {
                'category': str,       # 'Fraud/Unauthorized', 'Loan', 'KYC', 'Account Access', or 'General'
                'sentiment': str,      # 'Neutral', 'Urgent', 'Frustrated'
                'confidence': float    # Calibrated confidence score (0.0 to 1.0)
            }
        """
        pass

class BaseFraudClassifier(ABC):
    @abstractmethod
    def predict(self, transaction_id: str, query_text: str = "") -> dict:
        """
        Predict if a transaction is fraudulent based on transaction attributes
        and optional customer context / claim.

        Args:
            transaction_id: Unique transaction ID.
            query_text: Optional customer message / claim for contextual hybrid reasoning.

        Returns:
            dict: {
                'is_fraud': bool,      # True if fraud suspected
                'risk_level': str,     # 'High', 'Medium', or 'Low'
                'probability': float,  # Calibrated fraud probability (0.0 to 1.0)
                'reason': str,         # Explanation of risk indicators
                'suggested_action': str # Optional recommended action ('Block Card', 'Escalate Ticket', etc.)
            }
        """
        pass
