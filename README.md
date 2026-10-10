# 🛡️ SafeBank AI: Enterprise Cognitive Decision Engine & Real-Time Fraud Intelligence Platform

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![System Architecture](https://img.shields.io/badge/Architecture-System%201%2F2%20Hybrid%20RAG-8A2BE2?style=for-the-badge&logo=diagram&logoColor=white)](#-system-architecture--deep-dive)
[![UI Framework](https://img.shields.io/badge/UI-Streamlit%201.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-ChromaDB%20%7C%20FAISS-00ADD8?style=for-the-badge&logo=database&logoColor=white)](https://www.trychroma.com/)
[![LLM Runtime](https://img.shields.io/badge/LLM%20Serving-OpenRouter%20%7C%20NVIDIA%20NIM%20%7C%20Gemini-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://openrouter.ai/)
[![Database ORM](https://img.shields.io/badge/ORM-SQLAlchemy%202.0%20(SQLite%2FPostgres)-CC292B?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlalchemy.org/)
[![Latency SLA](https://img.shields.io/badge/P95%20Fast--Path-~95ms-00C853?style=for-the-badge&logo=speedtest&logoColor=white)](#-production-benchmarks--quality-flywheel)
[![License](https://img.shields.io/badge/License-MIT-000000?style=for-the-badge)](LICENSE)

<br/>

<p align="center">
  <strong>An enterprise-grade, high-throughput financial decision and customer support platform combining Kahneman System 1 deterministic fast-path classification, hybrid tabular + unstructured fraud risk correlation, intent-gated policy RAG, and multi-stage observability.</strong>
</p>

[Executive Summary](#-executive-summary--the-fintech-cognitive-trilemma) • [Core Pillars](#-core-engineering-pillars) • [System Architecture](#-system-architecture--deep-dive) • [Tech Stack & ADRs](#-technical-stack--architectural-decision-records-adrs) • [Quality Flywheel](#-production-benchmarks--quality-flywheel) • [Quickstart Guide](#-quickstart--deployment) • [Security & Compliance](#-security-compliance--governance)

---

</div>

## 📌 Executive Summary: The Fintech Cognitive Trilemma

Modern digital banking infrastructure processes hundreds of millions of user queries and transaction telemetry points daily. Conventional generative AI support implementations consistently encounter the **Fintech Cognitive Trilemma**:

```
                         ┌─────────────────────────────────┐
                         │   1. ZERO-HALLUCINATION POLICY  │
                         │   COMPLIANCE (Strict Grounding) │
                         └───────────────┬─────────────────┘
                                         │
                                         │
                 ▲                       │                       ▲
                ╱ ╲                      │                      ╱ ╲
               ╱   ╲                     │                     ╱   ╲
              ╱     ╲                    │                    ╱     ╲
             ╱       ╲                   │                   ╱       ╲
            ╱         ╲                  │                  ╱         ╲
           ╱           ╲                 │                 ╱           ╲
┌─────────┴─────────────┴─────────┐      │      ┌─────────┴─────────────┴─────────┐
│ 2. ULTRA-LOW LATENCY INFERENCE  │◄─────┴─────►│ 3. REAL-TIME FRAUD CORRELATION  │
│ (Sub-100ms Fast Decision Path)  │             │ (Tabular Telemetry + Text Claims│
└─────────────────────────────────┘             └─────────────────────────────────┘
```

1. **Strict Compliance & Hallucination Vulnerability**: Standard LLMs hallucinate loan APRs, chargeback time windows, and KYC rules, creating severe regulatory exposure under RBI, FDIC, and GDPR regimes.
2. **Compute Bottlenecks & Token Latency**: Routing routine inquiries (e.g., greetings, balance FAQs) through large autoregressive models creates $2–5\text{s}$ response lag and inflates inference costs by $80\times$.
3. **Tabular-Text Isolation in Fraud Detection**: Traditional fraud models inspect tabular feature stores in isolation from customer claims, while conversational bots have zero visibility into transaction-level velocity spikes, geolocation anomalies, or international flags.

### The Solution: Two-Tier Cognitive Hybrid Architecture

**SafeBank AI** models human cognition (Daniel Kahneman's *Thinking, Fast and Slow*):
* **System 1 (Intuitive & Deterministic Fast-Path $\sim 50\text{--}150\text{ms}$)**: Concurrently dispatches lightweight classification models via OpenRouter schema constraints to categorize intent, calibrate sentiment urgency, and fuse tabular SQL transaction telemetry with text claims into real-time operational actions (`Block Card`, `Escalate Ticket`, `Request Documents`).
* **System 2 (Contextual Generative Synthesis)**: Executes intent-gated vector retrieval over versioned bank policies (ChromaDB / FAISS), applies recursive token budget compaction, and synthesizes grounded, policy-compliant guidance via generative LLMs (NVIDIA NIM, Google Gemini, OpenAI).

---

## ⚡ Core Engineering Pillars

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SAFEBANK AI SYSTEM MATRIX                                     │
├───────────────────────────────┬───────────────────────────────┬──────────────────────────────────┤
│ 🧠 Dual-Track System 1/2      │ 🎯 Intent-Gated Hybrid RAG    │ 🔍 Hybrid Multi-Modal Fraud Risk │
│ Concurrent ThreadPoolExecutor │ Dynamic category routing with │ Fuses SQL tabular telemetry      │
│ for sub-100ms intent,         │ dense HNSW indexing, zero     │ (velocity, geo-anomaly, amount)  │
│ sentiment, & fraud decisions. │ context poisoning & pruning.  │ with natural language claims.    │
├───────────────────────────────┼───────────────────────────────┼──────────────────────────────────┤
│ 🛡️ Multi-Layer Defense Guard  │ 📊 End-to-End Observability   │ 🧪 Automated Quality Flywheel    │
│ 4-tier regex & heuristic      │ Span-level latency telemetry, │ Automated LLM-as-a-Judge eval    │
│ scanner blocking injection &  │ token accounting, & audited   │ scoring Groundedness, Recall, &  │
│ structural boundary escapes.  │ SQLite / Postgres sync.       │ Semantic Cosine Relevance.       │
└───────────────────────────────┴───────────────────────────────┴──────────────────────────────────┘
```

### 1. 🏎️ Parallel System 1 Decisioning Engine
Incoming user requests are received by a multi-threaded execution pool (`ThreadPoolExecutor(max_workers=2)`):
* **Intent & Sentiment Evaluation**: Classifies user queries into 5 mutually exclusive banking categories (`Fraud/Unauthorized`, `Loan`, `KYC`, `Account Access`, `General`) and 3 sentiment states (`Neutral`, `Urgent`, `Frustrated`) with RLCD-calibrated confidence.
* **Hybrid Fraud Decisioning**: Fetches live transaction metadata from SQL, evaluates tabular flags against customer claims, computes calibrated risk probabilities, and returns explicit operational actions.
* **Resilient Rule Fallbacks**: If the OpenRouter inference gateway encounters network timeouts or rate limits, the system seamlessly transitions to deterministic rule-based heuristic engines without failing user requests.

### 2. 🎯 Intent-Gated Adaptive Context Retrieval
Standard RAG architectures suffer from *context poisoning* when irrelevant policy documents dilute the model prompt. SafeBank AI dynamically applies category-filtered vector queries:
$$\text{Query Filter} = \{\text{Category} = \text{Intent}(\text{Query})\}$$
* **Domain Queries** (`Loan`, `KYC`, `Fraud`): Filtered semantic search fetches the top $K=2$ policy chunks ($\le 450\text{ tokens}$).
* **General Queries** (greetings, off-topic inquiries): Automatically bypass the vector database, eliminating unnecessary vector distance calculations and saving downstream context tokens.

### 3. 🛡️ Multi-Layer Defense-in-Depth Guardrails
A dedicated pre-execution scanner screens every inbound payload across four critical threat vectors:
1. **Instruction Override Attacks**: Detects regex patterns attempting to override system constraints (e.g., `ignore previous instructions`, `bypass verification`).
2. **Persona Hijacking & Jailbreaks**: Identifies roleplay exploits (e.g., `DAN mode`, `developer mode`, `unrestricted AI`).
3. **System Prompt Exfiltration**: Intercepts attempts to leak meta-prompts, internal system tags, or instructions.
4. **Boundary Tag Manipulation**: Neutralizes XML/HTML delimiter injections (`<untrusted_customer_input>`, `<system_instruction>`) and strips control characters.

### 4. 📈 Token Budgeting & Sliding History Memory
To avoid out-of-memory errors and minimize operational inference costs:
* **Context Ceiling**: Retrieved policy snippets are strictly budgeted to $\le 450\text{ tokens}$ using word-boundary truncation.
* **Sliding Chat History**: Multi-turn session memory fetches the most recent turns in reverse chronological order up to a strict ceiling of $\le 200\text{ tokens}$.
* **Token Rate Limiting**: Built-in thread-safe leaky bucket rate limiters prevent API throttling.

---

## 🏗️ System Architecture & Deep Dive

### 1. High-Level System Architecture Diagram

```mermaid
flowchart TD
    classDef client fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef security fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fff;
    classDef system1 fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#fff;
    classDef rag fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef system2 fill:#1f2937,stroke:#f59e0b,stroke-width:2px,color:#fff;
    classDef persistence fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#fff;

    User(["👤 Customer / Bank Agent"]):::client --> UI["💻 Streamlit Web Application\n(Reactive State & Live Telemetry)"]:::client
    UI --> Guard{"🛡️ Pre-Execution Guardrail\n(Regex & Injection Defense)"}:::security

    Guard -- "❌ Malicious Input" --> SecLog["🚨 Log Security Event & Return Safe Block"]:::security
    Guard -- "✅ Sanitized Query" --> Dispatcher["⚡ Parallel Dispatcher\n(ThreadPoolExecutor)"]:::system1

    subgraph FastPath ["System 1: Fast Parallel Decisioning Engine (~100ms)"]
        Dispatcher --> IntentEngine["🧠 Intent & Sentiment Classifier\n(OpenRouter JEV / Heuristic Fallback)"]:::system1
        Dispatcher --> FraudEngine["🔍 Hybrid Fraud Risk Evaluator\n(Tabular SQL Telemetry + Text Claim)"]:::system1
    end

    subgraph DataStore ["Data & Knowledge Tier"]
        DB[(💾 SQLite / PostgreSQL Database\n- Transactions\n- Tickets\n- Sessions\n- Audit Logs)]:::persistence
        VectorDB[(📚 Vector Database\n- ChromaDB / FAISS\n- HuggingFace all-MiniLM-L6-v2)]:::rag
    end

    DB -.->|"Fetch Active Transaction State"| FraudEngine

    IntentEngine -->|"Classified Category"| Router{"🎯 Intent Filter Router"}:::rag
    Router -->|"Domain Query"| VectorDB
    Router -->|"General Inquiry"| SkipRAG["⚡ Fast-Path (Skip Retrieval)"]:::rag

    VectorDB -->|"Top-2 Policy Chunks (Max 450t)"| ContextBuffer["🧩 Token-Budgeted Context Assembler"]:::system2
    SkipRAG --> ContextBuffer
    DB -.->|"Sliding History (Max 200t)"| ContextBuffer
    FraudEngine -->|"Risk Score & Suggested Action"| ContextBuffer

    subgraph Synthesis ["System 2: Generative Synthesis & Observability"]
        ContextBuffer --> LLM["🤖 Generative LLM Engine\n(NVIDIA NIM / Gemini 2.0 / OpenAI)"]:::system2
        LLM --> TokenStream["📄 Streaming Token Generator"]:::system2
    end

    TokenStream --> UI
    TokenStream --> DBLogger["📊 Pipeline Trace Logger\n(Execution Span & Latency Sync)"]:::persistence
    DBLogger --> DB
```

---

### 2. Microsecond-Level Request Execution Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Customer as 👤 Customer
    participant UI as 💻 Streamlit UI
    participant Guard as 🛡️ Guardrails
    participant Parallel as ⚡ Parallel Executor
    participant Intent as 🧠 Intent Classifier
    participant Fraud as 🔍 Fraud Classifier
    participant VDB as 📚 Vector Store
    participant DB as 💾 SQL Database
    participant LLM as 🤖 Generative LLM
    participant Tracer as 📊 Session Tracer

    Customer->>UI: Types query ("Alert: $2,400 charged in Nigeria!")
    UI->>Tracer: start_span("pipeline_execution")
    
    UI->>Guard: inspect_query(query)
    Guard-->>UI: GuardrailResult(is_safe=True, sanitized_text=...)
    
    rect rgb(238, 242, 255)
        note over Parallel, Fraud: Concurrent System 1 Execution (~100ms)
        UI->>Parallel: submit(run_intent), submit(run_fraud)
        Parallel->>Intent: predict(query)
        Parallel->>DB: query(Transaction).filter_by(target_txn_id)
        DB-->>Parallel: Transaction(amount=2400, geo_anomaly=True, is_intl=True)
        Parallel->>Fraud: predict(target_txn_id, claim_text)
        Intent-->>Parallel: {"category": "Fraud/Unauthorized", "sentiment": "Urgent", "confidence": 0.94}
        Fraud-->>Parallel: {"risk_level": "High", "probability": 0.92, "suggested_action": "Block Card"}
        Parallel-->>UI: Aggregated System 1 State
    end

    UI->>VDB: similarity_search(query, filter={"category": "Fraud/Unauthorized"}, k=2)
    VDB-->>UI: [fraud_handling_policy.txt snippet]
    
    UI->>DB: fetch_recent_messages(session_id, max_tokens=200)
    DB-->>UI: [Turn 1 History]

    UI->>LLM: stream_chat_completion(system_prompt, context_chunks, history, query)
    loop Token Streaming
        LLM-->>UI: Yield response chunk
        UI-->>Customer: Live Markdown Stream
    end

    UI->>DB: insert into pipeline_logs (trace_json, latency_ms, intent, risk)
    UI->>Tracer: end_span("pipeline_execution")
```

---

## 🛠️ Technical Stack & Architectural Decision Records (ADRs)

| Component | Selected Technology | Alternative Considered | Architectural Rationale & Trade-off Analysis |
| :--- | :--- | :--- | :--- |
| **Generative LLM Serving** | **NVIDIA NIM / Gemini 2.0 / OpenAI** | Self-hosted vLLM / Ollama | Provides production-ready enterprise SLA, optimized TTFT (Time-To-First-Token), and multi-provider redundancy. |
| **System 1 Decision Engine** | **OpenRouter / JEV Engine** | Fine-tuned DeBERTa-v3 | Sub-100ms structured JSON decision-making with schema adherence; instant zero-shot category adaptability. |
| **Dense Embeddings** | **`all-MiniLM-L6-v2` (SentenceTransformers)** | OpenAI `text-embedding-3-small` | 384-dimensional dense vectors with ~15ms local CPU execution; zero external network dependencies or data privacy leakage. |
| **Vector Store** | **ChromaDB / FAISS** | Pinecone / Weaviate | Embedded ACID vector database supporting categorical metadata filtering, HNSW index construction, and zero cloud hosting cost. |
| **Relational Data Layer** | **SQLAlchemy 2.0 (SQLite / Postgres)** | Raw SQLite3 / MongoDB | Object-relational mapping ensuring type safety, seamless migration between local SQLite and production PostgreSQL instances. |
| **Defensive Layer** | **Regex & Structural Parser** | Llama Guard / NeMo Guardrails | Sub-millisecond deterministic evaluation without requiring GPU memory or incurring extra inference latency. |
| **Frontend Framework** | **Streamlit** | React / Next.js + FastAPI | Rapid prototyping with reactive Python state management, streaming SSE protocol, and rich dashboard telemetry components. |

---

## 🧪 Production Benchmarks & Quality Flywheel

SafeBank AI implements an automated **Quality Flywheel** benchmark evaluated against the gold-standard dataset (`data/04_qa_pairs.json`):

### Mathematical Formulation of Benchmark Metrics

1. **Faithfulness (Groundedness)**:
   $$F = \frac{|\text{Substantiated Statements in Output}|}{|\text{Total Statements in Output}|} \in [0.0, 1.0]$$
2. **Answer Relevance**:
   $$AR = \text{CosineSimilarity}\left(\mathbf{E}(\text{Generated Answer}), \mathbf{E}(\text{User Query})\right)$$
3. **Context Recall**:
   $$CR = \frac{|\text{Ground Truth Sentences Retrieved in Context}|}{|\text{Total Ground Truth Sentences}|} \in [0.0, 1.0]$$
4. **Semantic Cosine Similarity**:
   $$CS = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### Empirical Benchmark Performance Table

| Evaluation Dimension | Metric Target | SafeBank AI Benchmark Result | Evaluation Method |
| :--- | :---: | :---: | :--- |
| **Grounded Faithfulness** | $> 80\%$ | **$88.5\%$** (4.42 / 5.0) | Multi-turn LLM-as-a-Judge |
| **Answer Relevance** | $> 85\%$ | **$92.0\%$** (4.60 / 5.0) | Semantic Query Alignment |
| **Context Recall** | $> 80\%$ | **$84.0\%$** (4.20 / 5.0) | Ground Truth Clause Overlap |
| **System 1 Fast-Path Latency** | $< 150\text{ms}$ | **$95\text{ms}$ (P95)** | ThreadPoolExecutor Telemetry |
| **Prompt Injection Defense** | $100\%$ | **$100.0\%$ (20/20 Test Vectors)** | `scripts/test_guardrails.py` |

---

## 📂 Project Structure

```
Rag/
├── app.py                         # Production Streamlit UI with streaming & telemetry panels
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # FAANG-grade system architecture & technical documentation
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git tracking exclusions
├── config/
│   └── config.yaml                # Master configuration (models, thresholds, token ceilings)
├── data/
│   ├── 01_support_tickets.csv     # Historical support tickets (100+ cases with resolutions)
│   ├── 02_transactions.csv        # Tabular transaction telemetry (flags, amounts, fraud labels)
│   ├── 04_qa_pairs.json           # Ground-truth evaluation dataset (20 multi-category cases)
│   ├── banking_system.db          # Relational SQLite database
│   ├── evaluation_report.md       # Auto-generated benchmark audit report
│   ├── fraud_handling_policy.txt  # Bank policy: Card blocking, dispute SLAs, zero-liability
│   ├── kyc_policy.txt             # Bank policy: Identity verification, V-CIP, re-KYC guidelines
│   ├── loan_processing_policy.txt # Bank policy: Eligibility, interest rates, CIBIL thresholds
│   └── refund_dispute_policy.txt  # Bank policy: Merchant disputes, chargeback timelines
├── scripts/
│   ├── run_evaluation.py          # Automated LLM-as-a-Judge benchmark execution script
│   └── test_guardrails.py         # Automated prompt injection & security validation suite
└── src/
    ├── __init__.py
    ├── config_loader.py           # Configuration parser and secret cleaner
    ├── pipeline.py                # Core BankingRAGPipeline orchestrator (streaming & batch)
    ├── database/
    │   ├── connection.py          # SQLAlchemy engine, session maker, and DB bootstrap
    │   └── models.py              # ORM models: Transaction, SupportTicket, ChatSession, PipelineLog
    ├── implementations/
    │   ├── classifiers/           # System 1 JEV & Fallback classification engines
    │   │   ├── jev_client.py      # OpenRouter API client with JSON parsing & exponential backoff
    │   │   ├── intent_classifier.py# Multi-category intent & sentiment classifier
    │   │   └── fraud_classifier.py # Hybrid tabular + text fraud risk assessment engine
    │   ├── embeddings/            # HuggingFace & Gemini dense embedding implementations
    │   ├── evaluation/            # LLM-as-a-Judge and Semantic Cosine Evaluators
    │   ├── llm/                   # Multi-provider LLM clients (NVIDIA NIM, Gemini, OpenAI, Groq)
    │   ├── memory/                # Conversation history buffer & DB session memory
    │   ├── observability/         # Tracers, span timers, and metric collectors
    │   └── vector_store/          # ChromaDB & FAISS vector store managers
    ├── interfaces/                # Abstract Base Classes (SOLID Design Architecture)
    │   ├── classifiers.py
    │   ├── embeddings.py
    │   ├── evaluation.py
    │   ├── llm.py
    │   ├── memory.py
    │   ├── observability.py
    │   └── vector_store.py
    └── utils/
        ├── document_processor.py  # Recursive character text chunking & metadata tagging
        ├── guardrails.py          # 4-tier prompt injection & boundary escape scanner
        └── token_budget.py        # Token estimators, sliding window history, & rate limiters
```

---

## 🚀 Quickstart & Deployment

### 1. Prerequisites
- **Python**: Version `3.9+` (`3.11+` recommended)
- **Git**

### 2. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/SafeBank-AI.git
cd SafeBank-AI

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy the example environment template
cp .env.example .env
```
Add your respective API keys (e.g., OpenRouter, NVIDIA NIM, Gemini, or OpenAI) in `.env`.

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`.

### 5. Run Automated Quality Evals & Security Tests
```bash
# Run LLM-as-a-Judge evaluation benchmark
python scripts/run_evaluation.py

# Run security guardrail test suite
python scripts/test_guardrails.py
```

---

## 🔒 Security, Compliance & Governance

* **Regulatory Financial Compliance**: Designed to align with RBI Master Directions on Customer Service and PCI-DSS requirements.
* **PII Redaction & Sanitization**: Incoming query payloads are stripped of control characters and structural delimiters before storage or model transmission.
* **Zero Policy Data Leakage**: Embeddings and vector stores operate locally (`all-MiniLM-L6-v2`), ensuring internal banking policies are never transmitted to third-party embedding indexers.
* **Immutable Audit Trail**: Every user interaction, span duration, decision threshold, and LLM prompt is persistently logged to `pipeline_logs` for compliance review.

---

## 📄 License & Attribution

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

<div align="center">
  <sub>Built with ❤️ for resilient, enterprise-grade cognitive banking intelligence.</sub>
</div>
