import streamlit as st
import numpy as np
import random
import time



st.set_page_config(page_title="Smart Dashboard",
                   layout="wide",
                   page_icon="📊",
                   initial_sidebar_state="expanded"
                   )
with st.sidebar:
    st.markdown("### History")
    st.markdown("📊 Q2 Sales Performance")
    st.markdown("📈 Revenue Analysis")
    st.markdown("🔄 Data Refresh")
    st.markdown("📋 Reports")

    st.markdown("---")
    with st.container(height=700):
        with st.form("db_query_form"):
            st.info("🧠 Query the vectorDB. The results will reflect the context retrieved by the RAG LLM.")

            db_query = st.text_area(
                "Enter a query...",
                height=100,
                placeholder="e.g., What happened on Dec 3rd regarding the steel plant project?"
            )

            query_submit = st.form_submit_button("🔍 Search")

        if query_submit:
            if not db_query.strip():
                st.warning("⚠️ Please enter a query before submitting.")
            else:
                st.success("✅ Query submitted!")
                st.markdown(f"**Your Query:** `{db_query}`")
st.markdown("Agentic Dashboard")
with st.form("agentic_config_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        code_interpreter_model = st.selectbox(
            "Select your code interpreter model",
            ("gemini-2.0-flash", "gemini-1.5-pro")
        )
        print(code_interpreter_model)

    with col2:
        rag_llm = st.selectbox(
            "Select your RAG LLM",
            ("llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b", "meta-llama/llama-guard-4-12b")
        )

    with col3:
        embedding_model = st.selectbox(
            "Select your Embedding model",
            ("sentence-transformers/all-MiniLM-L6-v2", "Cohere Embeddings")
        )

    files = st.file_uploader("Upload your Excel files..", accept_multiple_files=True, type=['.xlsx', '.pdf'])

    submitted = st.form_submit_button("submit")
if submitted:
    if not files:
        st.warning("⚠️ Please upload at least one `.xlsx` or `.pdf` file.")
    else:
        st.success("✅ Configuration submitted successfully!")
        # You can display selections for confirmation if needed
        st.markdown(f"**Selected Code Interpreter:** `{code_interpreter_model}`")
        st.markdown(f"**Selected RAG LLM:** `{rag_llm}`")
        st.markdown(f"**Selected Embedding Model:** `{embedding_model}`")
        st.markdown(f"**Number of Files Uploaded:** `{len(files)}`")

tab1, tab2, tab3 = st.tabs(["💬 QA Chatbot", "📊 Visualizations", "📝Project Details"])

# chatbot part


with tab1:
    st.markdown("### 💬 Ask your assistant")

    with st.container(height=600):
        def chat_stream(prompt):
            response = f'You said, "{prompt}" ...interesting.'
            for char in response:
                yield char
                time.sleep(0.005)


        def save_feedback(index):
            st.session_state.history[index]["feedback"] = st.session_state[f"feedback_{index}"]


        if "history" not in st.session_state:
            st.session_state.history = []

        for i, message in enumerate(st.session_state.history):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    feedback = message.get("feedback", None)
                    st.session_state[f"feedback_{i}"] = feedback
                    st.feedback(
                        "thumbs",
                        key=f"feedback_{i}",
                        disabled=feedback is not None,
                        on_change=save_feedback,
                        args=[i],
                    )

        if prompt := st.chat_input("Say something"):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                response = st.write_stream(chat_stream(prompt))
                st.feedback(
                    "thumbs",
                    key=f"feedback_{len(st.session_state.history)}",
                    on_change=save_feedback,
                    args=[len(st.session_state.history)],
                )
            st.session_state.history.append({"role": "assistant", "content": response})

with tab2:
    st.bar_chart(np.random.randn(30, 3))
