import numpy as np
from src.interfaces.evaluation import BaseEvaluator
from src.interfaces.llm import BaseLLMClient
from src.interfaces.embeddings import BaseEmbeddingModel

class LLMAsJudgeEvaluator(BaseEvaluator):
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.schema = {
            "type": "object",
            "properties": {
                "faithfulness": {
                    "type": "number",
                    "description": "Score from 1.0 to 5.0 indicating if the response is fully supported by the retrieved context, with no hallucinations."
                },
                "faithfulness_reasoning": {
                    "type": "string",
                    "description": "Short explanation for the faithfulness score."
                },
                "answer_relevance": {
                    "type": "number",
                    "description": "Score from 1.0 to 5.0 indicating if the response directly and fully answers the user query."
                },
                "answer_relevance_reasoning": {
                    "type": "string",
                    "description": "Short explanation for the answer relevance score."
                },
                "context_recall": {
                    "type": "number",
                    "description": "Score from 1.0 to 5.0 indicating how much of the ground truth information is captured in the retrieved context."
                },
                "context_recall_reasoning": {
                    "type": "string",
                    "description": "Short explanation for the context recall score."
                }
            },
            "required": [
                "faithfulness", "faithfulness_reasoning",
                "answer_relevance", "answer_relevance_reasoning",
                "context_recall", "context_recall_reasoning"
            ]
        }

    def evaluate(self, query: str, retrieved_context: str, generated_response: str, ground_truth: str = None) -> dict:
        from src.utils.token_budget import truncate_text_to_tokens

        gt_text = truncate_text_to_tokens(ground_truth, 150) if ground_truth else "Not Provided"
        ctx_budgeted = truncate_text_to_tokens(retrieved_context, 350)
        resp_budgeted = truncate_text_to_tokens(generated_response, 200)

        prompt = (
            "Grade banking RAG support QA (1.0 to 5.0 scale):\n"
            f"Query: \"{query}\"\n"
            f"Context: \"{ctx_budgeted}\"\n"
            f"Response: \"{resp_budgeted}\"\n"
            f"Ground Truth: \"{gt_text}\"\n\n"
            "Criteria:\n"
            "1. Faithfulness (1-5): Response grounded in context.\n"
            "2. Answer Relevance (1-5): Answers query directly.\n"
            "3. Context Recall (1-5): Context covers ground truth."
        )

        try:
            eval_results = self.llm_client.generate_structured_json(prompt, self.schema)
            # Normalize to 0-1 range for standardization, or keep 1-5
            return {
                'faithfulness': float(eval_results.get('faithfulness', 3.0)),
                'answer_relevance': float(eval_results.get('answer_relevance', 3.0)),
                'context_recall': float(eval_results.get('context_recall', 3.0)),
                'reasoning': {
                    'faithfulness': eval_results.get('faithfulness_reasoning', ''),
                    'answer_relevance': eval_results.get('answer_relevance_reasoning', ''),
                    'context_recall': eval_results.get('context_recall_reasoning', '')
                }
            }
        except Exception as e:
            print(f"Error in LLM-as-a-judge evaluation: {e}")
            return {
                'faithfulness': 1.0,
                'answer_relevance': 1.0,
                'context_recall': 1.0,
                'reasoning': {'error': str(e)}
            }

class SemanticEvaluator(BaseEvaluator):
    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model

    def evaluate(self, query: str, retrieved_context: str, generated_response: str, ground_truth: str = None) -> dict:
        if not ground_truth:
            return {
                'faithfulness': 0.0,
                'answer_relevance': 0.0,
                'context_recall': 0.0,
                'semantic_similarity': 0.0,
                'reasoning': {'info': "Ground truth not provided for semantic comparison."}
            }

        try:
            # Generate embeddings for response and ground truth
            resp_emb = np.array(self.embedding_model.embed_query(generated_response))
            gt_emb = np.array(self.embedding_model.embed_query(ground_truth))

            # Compute Cosine Similarity
            dot_product = np.dot(resp_emb, gt_emb)
            norm_resp = np.linalg.norm(resp_emb)
            norm_gt = np.linalg.norm(gt_emb)

            similarity = float(dot_product / (norm_resp * norm_gt))
            # Map similarity to 1-5 scale for equivalence with LLM judge
            mapped_score = float(round(1.0 + (similarity * 4.0), 2))

            return {
                'faithfulness': mapped_score,
                'answer_relevance': mapped_score,
                'context_recall': mapped_score,
                'semantic_similarity': similarity,
                'reasoning': {'similarity': f"Embedding similarity score: {similarity:.4f}"}
            }
        except Exception as e:
            print(f"Error in Semantic Evaluation: {e}")
            return {
                'faithfulness': 1.0,
                'answer_relevance': 1.0,
                'context_recall': 1.0,
                'reasoning': {'error': str(e)}
            }
