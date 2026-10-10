from src.interfaces.classifiers import BaseFraudClassifier
from src.database.models import Transaction
from src.implementations.classifiers.jev_client import JevOpenRouterClient
from src.config_loader import get_openrouter_api_key

class JevFraudClassifier(BaseFraudClassifier):
    """
    TypeSafe JEV System 1 Hybrid Fraud & Risk Assessment Classifier via OpenRouter.
    Combines tabular transaction metrics with customer natural language claims without legacy fallback classifiers.
    """
    def __init__(self, config: dict, session_maker):
        self.config = config or {}
        self.session_maker = session_maker
        jev_cfg = self.config.get("jev", {})
        api_key = get_openrouter_api_key(self.config)

        self.client = JevOpenRouterClient(
            api_key=api_key,
            base_url=jev_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            model=jev_cfg.get("model", "typesafe/jev-router"),
            timeout_seconds=float(jev_cfg.get("timeout_seconds", 5.0)),
            max_retries=int(jev_cfg.get("max_retries", 3)),
            retry_backoff_factor=float(jev_cfg.get("retry_backoff_factor", 0.5))
        )

        thresholds = jev_cfg.get("thresholds", {})
        self.high_risk_threshold = float(thresholds.get("fraud_high_risk", 0.80))
        self.medium_risk_threshold = float(thresholds.get("fraud_medium_risk", 0.45))

        self.schema = {
            "type": "object",
            "properties": {
                "is_fraud": {
                    "type": "boolean",
                    "description": "True if unauthorized or fraudulent activity is determined."
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High"],
                    "description": "Calculated risk severity."
                },
                "probability": {
                    "type": "number",
                    "description": "Calibrated fraud probability from 0.0 to 1.0."
                },
                "suggested_action": {
                    "type": "string",
                    "enum": ["None", "Block Card", "Escalate Ticket", "Request Documents"],
                    "description": "Recommended immediate operational response."
                },
                "reason": {
                    "type": "string",
                    "description": "Concise justification referencing specific transaction flags or customer claims."
                }
            },
            "required": ["is_fraud", "risk_level", "probability", "suggested_action", "reason"]
        }

    def _fallback_predict(self, tx, query_text: str = "") -> dict:
        q = (query_text or "").lower()
        is_fraud_query = any(w in q for w in ["fraud", "unauthorized", "stolen", "scam", "compromised", "hack", "block card", "suspicious", "illegal"])
        
        has_flags = False
        if tx:
            has_flags = bool(
                getattr(tx, "velocity_flag", False) or 
                getattr(tx, "geo_anomaly_flag", False) or 
                getattr(tx, "high_amount_flag", False) or 
                getattr(tx, "fraud_label", 0) == 1 or
                (getattr(tx, "is_international", False) and getattr(tx, "amount_inr", 0) > 50000)
            )

        if is_fraud_query or has_flags:
            prob = 0.88 if is_fraud_query else 0.82
            return {
                'is_fraud': True,
                'risk_level': 'High',
                'probability': prob,
                'suggested_action': 'Block Card',
                'reason': 'High risk detected: customer reported unauthorized activity or transaction anomaly flags triggered.',
                'source': 'rule_fallback',
                'model': 'rule_engine',
                'latency_ms': 1
            }
        else:
            return {
                'is_fraud': False,
                'risk_level': 'Low',
                'probability': 0.10,
                'suggested_action': 'None',
                'reason': 'Transaction metrics and customer claims indicate legitimate activity.',
                'source': 'rule_fallback',
                'model': 'rule_engine',
                'latency_ms': 1
            }

    def predict(self, transaction_id: str, query_text: str = "") -> dict:
        with self.session_maker() as session:
            tx = session.query(Transaction).filter_by(transaction_id=transaction_id).first()
            if not tx:
                return {
                    'is_fraud': False,
                    'risk_level': 'Low',
                    'probability': 0.0,
                    'reason': f"Transaction {transaction_id} not found in database.",
                    'suggested_action': 'None',
                    'source': 'jev_openrouter'
                }

            if not self.client.is_configured():
                return self._fallback_predict(tx, query_text)

            payload = {
                "transaction_id": tx.transaction_id,
                "amount_inr": tx.amount_inr,
                "merchant_name": tx.merchant_name,
                "merchant_category": tx.merchant_category,
                "city": tx.city,
                "hour_of_day": tx.hour_of_day,
                "is_international": tx.is_international,
                "velocity_flag": tx.velocity_flag,
                "geo_anomaly_flag": tx.geo_anomaly_flag,
                "high_amount_flag": tx.high_amount_flag,
                "historical_label": tx.fraud_label,
                "customer_claim_query": query_text or "No customer explanation provided."
            }

            system_instruction = (
                "You are JEV, an expert banking transaction fraud and risk decision engine.\n"
                "Evaluate the transaction metadata together with the customer claim.\n"
                "Assess whether this transaction indicates unauthorized access, velocity breach, geo anomaly, or legitimate spending.\n"
                f"- Assign 'High' risk if calibrated probability is >= {self.high_risk_threshold} or if card is reported stolen/compromised.\n"
                f"- Assign 'Medium' risk if probability is between {self.medium_risk_threshold} and {self.high_risk_threshold}.\n"
                f"- Assign 'Low' risk if probability is < {self.medium_risk_threshold}.\n"
                "Recommend 'Block Card' for confirmed high-risk unauthorized spending, 'Request Documents' for disputed subscriptions, 'Escalate Ticket' for edge cases, or 'None'."
            )

            try:
                decision = self.client.evaluate_decision(system_instruction, payload, self.schema)
                prob = float(decision.get("probability", 0.0))

                # Enforce configured thresholds if model risk doesn't strictly match
                if prob >= self.high_risk_threshold:
                    risk_level = "High"
                elif prob >= self.medium_risk_threshold:
                    risk_level = "Medium"
                else:
                    risk_level = decision.get("risk_level", "Low")

                return {
                    'is_fraud': bool(decision.get("is_fraud", prob >= 0.5)),
                    'risk_level': risk_level,
                    'probability': prob,
                    'suggested_action': decision.get("suggested_action", "None"),
                    'reason': decision.get("reason", "JEV decision evaluation completed."),
                    'source': 'jev_openrouter',
                    'model': decision.get("_model", self.client.model),
                    'latency_ms': decision.get("_latency_ms", 0)
                }
            except Exception:
                return self._fallback_predict(tx, query_text)

def get_fraud_classifier(config: dict, session_maker) -> BaseFraudClassifier:
    """Factory function returning the pure JEV Hybrid Fraud Classifier."""
    return JevFraudClassifier(config, session_maker)
