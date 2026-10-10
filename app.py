import os
import time
import json
import pandas as pd
import streamlit as st

# Set page config FIRST, before any other imports that might do streamlit stuff
st.set_page_config(
    page_title="SafeBank AI Support & Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config_loader import (
    load_config,
    is_api_key_set,
    get_api_key_env_var_name,
    is_openrouter_api_key_set
)
from src.database.connection import get_engine, get_session_maker, init_db
from src.database.models import Transaction, SupportTicket
from src.implementations.embeddings.models import get_embedding_model
from src.implementations.vector_store.stores import get_vector_store
from src.implementations.llm.clients import get_llm_client
from src.implementations.classifiers.intent_classifier import get_intent_classifier
from src.implementations.classifiers.fraud_classifier import get_fraud_classifier
from src.implementations.memory.history import SQLAlchemyMemory
from src.implementations.observability.tracers import SessionTracer
from src.implementations.evaluation.evaluators import LLMAsJudgeEvaluator, SemanticEvaluator
from src.pipeline import BankingRAGPipeline

# Custom CSS for rich aesthetics and modern layout (glassmorphism/gradients)
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Header & Gradient Accents */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin-top: 0.5rem;
    }

    /* Badges & Metrics */
    .metric-badge {
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-purple {
        background-color: rgba(168, 85, 247, 0.2);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }

    /* Collapsible Console Box */
    .developer-console {
        background-color: #0f172a;
        color: #38bdf8;
        padding: 1.5rem;
        border-radius: 12px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
        border: 1px solid #334155;
    }

    /* Chat bubbles styling */
    .chat-user {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff !important;
        padding: 1rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 1rem;
        text-align: right;
        border: 1px solid #3b82f6;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
        line-height: 1.5;
        font-size: 0.98rem;
    }
    .chat-bot {
        background: #1e293b;
        color: #f8fafc !important;
        padding: 1.2rem 1.3rem;
        border-radius: 4px 16px 16px 16px;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        line-height: 1.6;
        font-size: 0.98rem;
    }
    .chat-bot b, .chat-bot strong {
        color: #ffffff !important;
    }
    .chat-bot code {
        background-color: #0f172a !important;
        color: #38bdf8 !important;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "selected_customer" not in st.session_state:
    st.session_state.selected_customer = "CUST103161"
if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = None
if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = None

# Initialize pipeline caching to speed up loads
@st.cache_resource
def init_pipeline_resources():
    config = load_config()

    # Initialize SQLite Database
    engine = get_engine(config["database"]["connection_string"])
    init_db(config["database"]["connection_string"])
    session_maker = get_session_maker(engine)

    # Initialize Core Classes
    embedding_model = get_embedding_model(config)
    vector_store = get_vector_store(config, embedding_model)

    # Check for API keys
    api_key_name = get_api_key_env_var_name(config)
    api_key_set = is_api_key_set(config)
    openrouter_key_set = is_openrouter_api_key_set(config)

    try:
        llm_client = get_llm_client(config)
    except Exception as e:
        class FallbackLLM:
            def generate_response(self, p, s=None): return "No API key configured."
            def generate_structured_json(self, p, s):
                return {
                    "response": f"{api_key_name} is missing. Please set {api_key_name} in your .env file and restart the application.",
                    "risk_level": "Low",
                    "suggested_action": "None"
                }
        llm_client = FallbackLLM()

    intent_classifier = get_intent_classifier(config, llm_client)
    fraud_classifier = get_fraud_classifier(config, session_maker)
    memory = SQLAlchemyMemory(session_maker, default_customer_id=st.session_state.selected_customer)
    tracer = SessionTracer()

    pipeline = BankingRAGPipeline(
        config=config,
        session_maker=session_maker,
        llm_client=llm_client,
        vector_store=vector_store,
        intent_classifier=intent_classifier,
        fraud_classifier=fraud_classifier,
        memory=memory,
        tracer=tracer
    )

    # Set up evaluators
    semantic_evaluator = SemanticEvaluator(embedding_model)
    llm_judge = None
    if api_key_set:
        try:
            llm_judge = LLMAsJudgeEvaluator(llm_client)
        except:
            pass

    return config, session_maker, pipeline, semantic_evaluator, llm_judge, api_key_set, openrouter_key_set

# Load objects
config, session_maker, pipeline, semantic_evaluator, llm_judge, api_key_set, openrouter_key_set = init_pipeline_resources()
api_key_name = get_api_key_env_var_name(config)

# Update memory customer ID dynamically on selection change
pipeline.memory.default_customer_id = st.session_state.selected_customer

# ----------------- UI HEADER -----------------
llm_status_text = "Connected" if api_key_set else "API Key Missing"
llm_status_color = "#4ade80" if api_key_set else "#f87171"

jev_status_text = "Active (OpenRouter)" if openrouter_key_set else "API Key Missing"
jev_status_color = "#4ade80" if openrouter_key_set else "#f87171"

st.markdown(f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="header-title">🛡️ SafeBank AI Support & Fraud Intelligence</h1>
            <p class="header-subtitle">Production-grade Hybrid RAG + TypeSafe JEV System 1 Decision Engine with Live Traceability</p>
        </div>
        <div style="text-align: right; background: rgba(255,255,255,0.1); padding: 12px 20px; border-radius: 10px;">
            <div style="font-size: 0.85rem; margin-bottom: 4px;">
                <span style="font-weight: 600;">JEV ENGINE:</span>
                <span style="color: {jev_status_color}; font-weight: bold;">{jev_status_text}</span>
            </div>
            <div style="font-size: 0.85rem;">
                <span style="font-weight: 600;">RAG LLM:</span>
                <span style="color: {llm_status_color}; font-weight: bold;">{llm_status_text}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR: CUSTOMER PROFILE -----------------
with st.sidebar:
    st.markdown("### 👤 Customer Selector (Simulated)")

    with session_maker() as session:
        unique_customers = session.query(Transaction.account_id).distinct().limit(10).all()
        customer_options = []
        for row in unique_customers:
            cust_num = "".join(filter(str.isdigit, row[0]))
            customer_options.append(f"CUST{cust_num}")

        customer_options = sorted(list(set(customer_options))) if customer_options else ["CUST103161"]

    selected_cust = st.selectbox(
        "Choose Customer Persona to Simulate:",
        customer_options,
        index=customer_options.index(st.session_state.selected_customer) if st.session_state.selected_customer in customer_options else 0
    )

    if selected_cust != st.session_state.selected_customer:
        st.session_state.selected_customer = selected_cust
        st.session_state.session_id = f"session_{selected_cust}_{int(time.time())}"
        st.session_state.pipeline_output = None
        st.session_state.eval_scores = None
        st.rerun()

    cust_num = "".join(filter(str.isdigit, st.session_state.selected_customer))
    acc_id = f"ACC{cust_num}"

    with session_maker() as session:
        txs = session.query(Transaction).filter_by(account_id=acc_id).order_by(Transaction.timestamp.desc()).all()
        tickets = session.query(SupportTicket).filter_by(customer_id=st.session_state.selected_customer).all()

    st.markdown("---")
    st.markdown(f"**Account ID:** `{acc_id}`")
    st.markdown(f"**Total Transactions:** {len(txs)}")
    st.markdown(f"**Past Support Tickets:** {len(tickets)}")

    st.markdown("---")
    st.markdown("### 💳 Flagged/Select Transaction")
    st.write("Select a transaction ID if disputing a specific charge:")

    tx_ids = ["None"] + [tx.transaction_id for tx in txs]
    selected_tx_id = st.selectbox("Disputed Transaction:", tx_ids)

    st.markdown("---")
    st.markdown("### 📜 Recent Transactions")
    if txs:
        tx_data = []
        for t in txs[:5]:
            status = "⚠️ Fraud Flagged" if t.fraud_label == 1 else "Legitimate"
            tx_data.append({
                "ID": t.transaction_id,
                "Amount": f"₹{t.amount_inr:,.2f}",
                "Merchant": t.merchant_name,
                "Type": t.transaction_type,
                "Status": status
            })
        st.dataframe(pd.DataFrame(tx_data), hide_index=True)
    else:
        st.write("No transaction records found.")

# ----------------- MAIN UI LAYOUT -----------------
tab_chat, tab_details = st.tabs(["💬 SafeBank Support Chat", "🧠 AI Cognitive Engine Dashboard"])

# ----------------- TAB 1: CHATBOT INTERFACE -----------------
with tab_chat:
    st.markdown("### 💬 SafeBank Support Chat")

    history = pipeline.memory.get_history(st.session_state.session_id, limit=20)

    chat_container = st.container(height=400)
    with chat_container:
        if not history:
            st.markdown(
                f'<div class="chat-bot">Hello! I am <b>SafeBank AI Assistant</b>. '
                f'I am simulating support for customer <b>{st.session_state.selected_customer}</b> (Account: <code>{acc_id}</code>). '
                f'How can I assist you with your banking policies, KYC, loans, or transactions today?</div>',
                unsafe_allow_html=True
            )
        else:
            for msg in history:
                if msg['role'] == 'user':
                    st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    user_query = st.chat_input("Type your message here...")

    if user_query:
        tx_param = None if selected_tx_id == "None" else selected_tx_id
        with chat_container:
            st.markdown(f'<div class="chat-user">{user_query}</div>', unsafe_allow_html=True)
            bot_placeholder = st.empty()

            accumulated_response = ""
            final_output = None

            with st.spinner("Analyzing intent & risk in parallel via JEV..."):
                gen = pipeline.stream_run(st.session_state.session_id, user_query, transaction_id=tx_param)
                for item in gen:
                    if item["type"] == "token":
                        accumulated_response += item["content"]
                        bot_placeholder.markdown(f'<div class="chat-bot">{accumulated_response}▌</div>', unsafe_allow_html=True)
                    elif item["type"] == "final":
                        final_output = item["output"]

            bot_placeholder.markdown(f'<div class="chat-bot">{accumulated_response}</div>', unsafe_allow_html=True)

        if final_output:
            st.session_state.pipeline_output = final_output
            st.session_state.eval_scores = None
            st.rerun()

    if st.button("Reset Session History"):
        pipeline.memory.clear(st.session_state.session_id)
        st.session_state.pipeline_output = None
        st.session_state.eval_scores = None
        st.rerun()

# ----------------- TAB 2: ANALYTICS & COGNITIVE CONSOLE -----------------
with tab_details:
    st.markdown("### 🧠 AI Cognitive Engine Dashboard")

    if st.session_state.pipeline_output is None:
        st.info("Submit a chat query to view JEV's System 1 decisions, policy context, and live execution traces in real-time.")
    else:
        out = st.session_state.pipeline_output

        st.markdown("#### Real-time JEV Classification & Telemetry")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Detected Intent", out["intent"])
            st.markdown(f'<span class="metric-badge badge-blue">{out["intent"]}</span>', unsafe_allow_html=True)
        with c2:
            st.metric("Customer Sentiment", out["sentiment"])
            badge_style = "badge-medium" if out["sentiment"] in ["Urgent", "Frustrated"] else "badge-low"
            st.markdown(f'<span class="metric-badge {badge_style}">{out["sentiment"]}</span>', unsafe_allow_html=True)
        with c3:
            conf_val = out.get("intent_confidence", 0.85)
            st.metric("Decision Confidence", f"{conf_val:.1%}")
            src_tag = out.get("classifier_source", "jev_openrouter")
            st.markdown(f'<span class="metric-badge badge-purple">{src_tag}</span>', unsafe_allow_html=True)
        with c4:
            st.metric("Calculated Risk", out["risk_level"])
            risk_badge = "badge-high" if out["risk_level"] == "High" else ("badge-medium" if out["risk_level"] == "Medium" else "badge-low")
            st.markdown(f'<span class="metric-badge {risk_badge}">{out["risk_level"]} Risk</span>', unsafe_allow_html=True)

        st.write("---")

        # Triggered Action Layer
        st.markdown("#### ⚡ Triggered Action Layer")
        action = out["suggested_action"]
        if action == "None":
            st.success("✅ **System Action:** No intervention required.")
        else:
            action_icon = "🚨" if action in ["Block Card", "Escalate Ticket"] else "📝"
            st.warning(f"{action_icon} **Recommended Action:** `{action}` triggered.")

        st.write("---")

        # Sub-tabs
        tab_trace, tab_retrieval, tab_fraud_jev, tab_eval = st.tabs([
            "🔍 Execution Trace",
            "📚 Retrieved Policies",
            "🛡️ JEV Fraud & Risk Engine",
            "⚖️ Live Evaluation"
        ])

        # TAB 1: TRACING
        with tab_trace:
            token_usage = out.get("token_usage", {})
            total_tok = token_usage.get("total_tokens", 0)
            prompt_tok = token_usage.get("prompt_tokens", 0)
            comp_tok = token_usage.get("completion_tokens", 0)

            c_lat, c_tok = st.columns(2)
            c_lat.metric("Pipeline Latency", f"{out['latency_ms']} ms")
            c_tok.metric("Tokens (Budgeted)", f"{total_tok} tok", f"In: {prompt_tok} | Out: {comp_tok}")
            st.write("Step-by-step execution metrics:")

            for span in out["traces"]:
                status_icon = "❌" if span.get("error") else "✅"
                name = span["name"].upper().replace("_", " ")
                duration = span["duration_ms"]

                inputs = span.get("inputs") or {}
                outputs = span.get("outputs") or {}

                with st.expander(f"{status_icon} {name} — {duration}ms"):
                    st.write("**Inputs:**")
                    st.json(inputs)
                    st.write("**Outputs:**")
                    st.json(outputs)

        # TAB 2: POLICIES
        with tab_retrieval:
            ret_docs = out["retrieved_context"]
            st.markdown(f"**Retrieved Document Chunks:** {len(ret_docs)}")

            if not ret_docs:
                st.write("No external knowledge context retrieved.")
            else:
                for idx, doc in enumerate(ret_docs):
                    meta = doc["metadata"]
                    st.markdown(f"**Chunk {idx+1}** | Source: `{meta.get('source')}` | Score: `{doc.get('score', 0.0):.4f}`")
                    st.info(doc["text"])

        # TAB 3: JEV FRAUD ENGINE
        with tab_fraud_jev:
            fraud = out["fraud_details"]
            f_prob = fraud.get('probability', 0.0)
            st.markdown(f"**Calibrated Fraud Probability (RLCD):** `{f_prob:.2%}`")
            st.progress(min(max(f_prob, 0.0), 1.0))

            st.markdown(f"**Assessed Risk Level:** `{fraud.get('risk_level')}`")
            st.markdown(f"**JEV Evaluation Source:** `{fraud.get('source', 'jev_openrouter')}`")
            if 'model' in fraud:
                st.markdown(f"**Model Identifier:** `{fraud.get('model')}`")
            st.markdown(f"**Decision Reasoning (Text + Tabular Context):**")
            st.info(fraud.get('reason', 'No reasoning details provided.'))

            if "transaction_details" in fraud:
                st.write("**Analyzed Transaction Metadata:**")
                st.json(fraud["transaction_details"])
            else:
                st.write("No specific transaction was attached for risk analysis.")

        # TAB 4: EVALUATION
        with tab_eval:
            st.markdown("#### Real-time Generation Quality Metrics")
            st.write("Evaluate the system's generated response quality:")

            if not api_key_set:
                st.info(f"LLM-as-a-judge requires a valid {api_key_name}.")

            if st.button("Run Real-time Evaluator"):
                with st.spinner("Calculating quality metrics..."):
                    ground_truth = None
                    qa_path = "data/04_qa_pairs.json"
                    if os.path.exists(qa_path):
                        with open(qa_path, "r", encoding="utf-8") as f:
                            qa_pairs = json.load(f)
                        for qa in qa_pairs:
                            q_clean = qa["question"].lower()
                            user_q_clean = out["query"].lower()
                            if q_clean[:20] in user_q_clean or user_q_clean[:20] in q_clean:
                                ground_truth = qa["answer"]
                                break
                        if not ground_truth:
                            try:
                                import numpy as np
                                user_q_emb = np.array(semantic_evaluator.embedding_model.embed_query(out["query"]))
                                best_score = 0.0
                                for qa in qa_pairs:
                                    qa_emb = np.array(semantic_evaluator.embedding_model.embed_query(qa["question"]))
                                    dot_product = np.dot(user_q_emb, qa_emb)
                                    norm_user = np.linalg.norm(user_q_emb)
                                    norm_qa = np.linalg.norm(qa_emb)
                                    if norm_user > 0 and norm_qa > 0:
                                        score = float(dot_product / (norm_user * norm_qa))
                                        if score > best_score and score > 0.70:
                                            best_score = score
                                            ground_truth = qa["answer"]
                            except Exception as e:
                                print(f"Error in semantic QA matching: {e}")

                    retrieved_str = "\n\n".join([doc['text'] for doc in out["retrieved_context"]])

                    sem_res = semantic_evaluator.evaluate(
                        query=out["query"],
                        retrieved_context=retrieved_str,
                        generated_response=out["response"],
                        ground_truth=ground_truth
                    )

                    judge_res = None
                    if llm_judge and api_key_set:
                        try:
                            judge_res = llm_judge.evaluate(
                                query=out["query"],
                                retrieved_context=retrieved_str,
                                generated_response=out["response"],
                                ground_truth=ground_truth
                            )
                        except Exception as e:
                            st.error(f"LLM Judge error: {e}")

                    st.session_state.eval_scores = {
                        "semantic": sem_res,
                        "judge": judge_res,
                        "has_gt": ground_truth is not None
                    }

            if st.session_state.eval_scores:
                scores = st.session_state.eval_scores
                sem = scores["semantic"]
                similarity = sem.get("semantic_similarity", 0.0)
                st.metric("Semantic Similarity (Cosine)", f"{similarity:.4f}")

                if not scores.get("has_gt"):
                    st.info("ℹ️ Note: This query did not match any reference Q&A pairs in the dataset. Semantic similarity is set to 0.0000 because no ground truth reference answer is available for comparison.")

                if scores["judge"]:
                    judge = scores["judge"]
                    ec1, ec2, ec3 = st.columns(3)
                    with ec1:
                        st.metric("Faithfulness Score", f"{judge['faithfulness']:.1f}/5.0")
                        st.caption(f"Reasoning: {judge['reasoning'].get('faithfulness', '')}")
                    with ec2:
                        st.metric("Answer Relevance", f"{judge['answer_relevance']:.1f}/5.0")
                        st.caption(f"Reasoning: {judge['reasoning'].get('answer_relevance', '')}")
                    with ec3:
                        st.metric("Context Recall", f"{judge['context_recall']:.1f}/5.0")
                        st.caption(f"Reasoning: {judge['reasoning'].get('context_recall', '')}")

                    if judge['faithfulness'] < 3.5:
                        st.warning("⚠️ Low Faithfulness detected: Generated response may contain hallucinations not backed by policy documents.")
                    if judge['context_recall'] < 3.5:
                        st.info("ℹ️ Low Context Recall: The retriever may have missed key policy guidelines mentioned in reference answers.")
                else:
                    st.info(f"LLM-as-a-judge scores not generated. Set {api_key_name} to see full evaluation.")
