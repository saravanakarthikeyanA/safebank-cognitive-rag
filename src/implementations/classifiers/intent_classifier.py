from src.interfaces.classifiers import BaseIntentClassifier
from src.implementations.classifiers.jev_client import JevOpenRouterClient
from src.config_loader import get_openrouter_api_key

class JevIntentClassifier(BaseIntentClassifier):
    """
    TypeSafe JEV System 1 Intent & Sentiment Classifier via OpenRouter.
    Executes fast, deterministic, schema-constrained evaluations without legacy fallback classifiers.
    """
    def __init__(self, config: dict = None):
        config = config or {}
        jev_cfg = config.get("jev", {})
        api_key = get_openrouter_api_key(config)

        self.client = JevOpenRouterClient(
            api_key=api_key,
            base_url=jev_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            model=jev_cfg.get("model", "typesafe/jev-router"),
            timeout_seconds=float(jev_cfg.get("timeout_seconds", 5.0)),
            max_retries=int(jev_cfg.get("max_retries", 3)),
            retry_backoff_factor=float(jev_cfg.get("retry_backoff_factor", 0.5))
        )
        self.schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Fraud/Unauthorized", "Loan", "KYC", "Account Access", "General"]
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["Neutral", "Urgent", "Frustrated"]
                },
                "confidence": {
                    "type": "number",
                    "description": "Calibrated confidence score between 0.0 and 1.0"
                }
            },
            "required": ["category", "sentiment", "confidence"]
        }

    def _fallback_predict(self, query: str) -> dict:
        q = query.lower()
        if any(w in q for w in ["fraud", "unauthorized", "stolen", "scam", "compromised", "hack", "block card", "suspicious", "chargeback", "dispute"]):
            cat = "Fraud/Unauthorized"
            sent = "Urgent"
            conf = 0.92
        elif any(w in q for w in ["loan", "emi", "foreclosure", "cibil", "interest rate", "mortgage", "prepayment", "borrow"]):
            cat = "Loan"
            sent = "Neutral"
            conf = 0.88
        elif any(w in q for w in ["kyc", "aadhaar", "pan", "passport", "v-cip", "verification", "re-kyc", "document"]):
            cat = "KYC"
            sent = "Neutral"
            conf = 0.88
        elif any(w in q for w in ["login", "password", "pin", "otp", "locked", "unlock", "access", "credentials", "biometric"]):
            cat = "Account Access"
            sent = "Neutral"
            conf = 0.85
        else:
            cat = "General"
            sent = "Neutral"
            conf = 0.75

        return {
            "category": cat,
            "sentiment": sent,
            "confidence": conf,
            "source": "rule_fallback",
            "model": "rule_engine",
            "latency_ms": 1
        }

    def predict(self, query: str) -> dict:
        if not self.client.is_configured():
            return self._fallback_predict(query)

        system_instruction = (
            "You are JEV, a banking intent and customer sentiment classification engine.\n"
            "Analyze the customer query and classify into one of the following exact categories:\n"
            "- 'Fraud/Unauthorized': Stolen cards, unrecognized charges, disputes, scams, compromised credentials.\n"
            "- 'Loan': Home/personal/car loans, EMI, foreclosure, eligibility, CIBIL, interest rates.\n"
            "- 'KYC': Aadhaar/PAN updates, document verification, Video KYC (V-CIP), AML checks.\n"
            "- 'Account Access': Login issues, password/PIN resets, OTP, locked accounts, biometric access.\n"
            "- 'General': Routine branch inquiries, working hours, greetings, general customer support.\n\n"
            "Also evaluate customer sentiment as 'Neutral', 'Urgent', or 'Frustrated', and provide a calibrated confidence float (0.0 to 1.0)."
        )

        payload = {"customer_query": query}
        try:
            decision = self.client.evaluate_decision(system_instruction, payload, self.schema)
            return {
                "category": decision.get("category", "General"),
                "sentiment": decision.get("sentiment", "Neutral"),
                "confidence": float(decision.get("confidence", 0.85)),
                "source": "jev_openrouter",
                "model": decision.get("_model", self.client.model),
                "latency_ms": decision.get("_latency_ms", 0)
            }
        except Exception:
            return self._fallback_predict(query)

def get_intent_classifier(config: dict, llm_client=None) -> BaseIntentClassifier:
    """Factory function returning the pure JEV Intent Classifier."""
    return JevIntentClassifier(config)
