# Banking RAG System Evaluation Report

Generated on: 2026-07-20 13:04:14

## Overall Metrics
- **Total Test Cases**: 20
- **Average Latency**: 15589.99 ms
- **Average Semantic Similarity (Cosine)**: 0.6127 (1-5 equivalent: 3.45)
- **Average Faithfulness (LLM Judge)**: 2.75 / 5.0
- **Average Answer Relevance (LLM Judge)**: 3.08 / 5.0
- **Average Context Recall (LLM Judge)**: 3.12 / 5.0

## Test Case Detail Breakdown

| ID | Category | Query | Expected Risk | Actual Risk | Sem. Sim | Faithfulness | Relevance | Recall |
|---|---|---|---|---|---|---|---|---|
| QA001 | Fraud | I noticed an unauthorized tran... | High | Low | 0.59 | 4.0 | 4.0 | 5.0 |
| QA002 | Fraud | How long does fraud investigat... | Medium | Low | 0.39 | 4.0 | 4.0 | 5.0 |
| QA003 | Fraud | Someone used my OTP to transfe... | High | Low | 0.55 | 1.0 | 3.0 | 1.0 |
| QA004 | Fraud | Multiple small transactions ar... | High | Low | 0.44 | 3.0 | 4.0 | 4.0 |
| QA005 | Fraud | I received an OTP I didn't req... | High | Low | 0.70 | 1.0 | 2.0 | 1.0 |
| QA006 | Loan | What is the minimum CIBIL scor... | Low | Low | 0.86 | 5.0 | 4.0 | 4.0 |
| QA007 | Loan | How long will my home loan tak... | Low | Low | 0.90 | 4.0 | 4.5 | 4.5 |
| QA008 | Loan | Can I foreclose my floating ra... | Low | Low | 0.59 | 2.0 | 2.0 | 3.0 |
| QA009 | Loan | My loan application was reject... | Low | Low | 0.69 | 4.0 | 4.0 | 4.5 |
| QA010 | Loan | I lost my job. Can I pause my ... | High | Low | 0.41 | 2.0 | 2.5 | 3.0 |
| QA011 | KYC | Why is my account restricted? ... | Medium | Low | 0.61 | 4.0 | 3.5 | 4.5 |
| QA012 | KYC | My Aadhaar name and bank name ... | Medium | Low | 0.46 | 3.0 | 3.0 | 2.0 |
| QA013 | KYC | What documents do I need for K... | Low | Low | 0.67 | 4.0 | 4.0 | 4.0 |
| QA014 | KYC | How often do I need to update ... | Low | Low | 0.77 | 4.0 | 4.0 | 5.0 |
| QA015 | Account Access | I'm locked out of my net banki... | Low | Low | 0.68 | 1.0 | 3.0 | 1.0 |
| QA016 | Account Access | My salary hasn't been credited... | Low | Low | 0.56 | 2.0 | 3.0 | 4.0 |
| QA017 | Fraud | I got a call from someone clai... | High | Low | 0.77 | 1.0 | 1.0 | 1.0 |
| QA018 | Account Access | Can I open a joint account wit... | Low | Low | 0.54 | 1.0 | 1.0 | 1.0 |
| QA019 | Loan | What is the maximum education ... | Low | Low | 0.86 | 4.0 | 4.0 | 4.0 |
| QA020 | KYC | My video KYC keeps failing. Wh... | Low | Medium | 0.22 | 1.0 | 1.0 | 1.0 |
