"""
app.py — Streamlit UI for Ask My Docs.

Run: streamlit run ui/app.py
"""
import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    collection = st.text_input("Collection", value="ask_my_docs")
    reranker = st.selectbox("Reranker", ["local", "cohere"])
    st.divider()
    st.subheader("📥 Ingest Documents")
    docs_dir = st.text_input("Docs directory", value="data/raw")
    if st.button("Run Ingestion", type="secondary"):
        with st.spinner("Ingesting..."):
            resp = requests.post(f"{API_BASE}/ingest", json={
                "docs_dir": docs_dir,
                "collection": collection,
            })
            if resp.ok:
                data = resp.json()
                st.success(f"Ingested {data['chunks']} chunks in {data['elapsed_seconds']:.1f}s")
            else:
                st.error(resp.text)

# --- Main ---
st.title("📚 Ask My Docs")
st.caption("Production RAG · Hybrid Retrieval · Citation Enforcement")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s)"):
                for c in msg["citations"]:
                    st.markdown(f"**[{c['index']}]** `{c['source']}`"
                                + (f" · page {c['page']}" if c.get('page') else "")
                                + f" · relevance {c['relevance_score']:.2f}")
                    st.caption(c["excerpt"])

# Input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                resp = requests.post(f"{API_BASE}/query", json={
                    "question": question,
                    "collection": collection,
                    "reranker": reranker,
                })
                if resp.ok:
                    data = resp.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    timings = data["latency_ms"]
                    tokens = data["token_usage"]

                    st.markdown(answer)

                    # Latency + token info
                    cols = st.columns(4)
                    cols[0].metric("Total", f"{timings.get('total_ms', 0):.0f}ms")
                    cols[1].metric("Retrieval", f"{timings.get('retrieval_ms', 0):.0f}ms")
                    cols[2].metric("Generation", f"{timings.get('generation_ms', 0):.0f}ms")
                    cols[3].metric("Tokens", f"{tokens.get('input', 0) + tokens.get('output', 0)}")

                    if citations:
                        with st.expander(f"📎 {len(citations)} source(s) cited"):
                            for c in citations:
                                st.markdown(f"**[{c['index']}]** `{c['source']}`"
                                            + (f" · page {c['page']}" if c.get('page') else "")
                                            + f" · relevance {c['relevance_score']:.2f}")
                                st.caption(c["excerpt"])

                    if not data.get("citations_valid"):
                        st.warning("⚠️ Some citations could not be verified against sources.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    })
                else:
                    st.error(f"API error: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the server is running: `make api`")
