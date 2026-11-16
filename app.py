
import sys
import time
from pathlib import Path

import streamlit as st

# --------------------------------
# PROJECT PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SRC_DIR))

from retriever import InsuranceRetriever
from gemini_generator import GeminiGenerator


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Insurance Policy Assistant",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Policy Assistant")

st.caption(
    "Your AI-powered assistant for insurance coverage, "
    "claims, premiums, exclusions and riders."
)

st.divider()


# --------------------------------
# LOAD RAG COMPONENTS
# --------------------------------

@st.cache_resource
def load_components():

    retriever = InsuranceRetriever()
    generator = GeminiGenerator()

    return retriever, generator


try:

    with st.spinner("Initializing Insurance Assistant..."):

        retriever, generator = load_components()

except Exception as e:

    st.error("Unable to initialize the Insurance Assistant.")
    st.exception(e)
    st.stop()


# --------------------------------
# SESSION STATE
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

        # Document source selection

    
    st.subheader("Document Source")

    st.info("Using JSON insurance policy documents only.")


    st.divider()

    retrieval_only = st.checkbox(
        "Retrieval testing mode (Skip Gemini)",
        value=True
    )

    st.write("You can ask questions about:")

    st.markdown("""
    - Hospitalization coverage
    - Policy exclusions
    - Waiting periods
    - Cashless claims
    - Reimbursement claims
    - Premium calculations
    - Policy renewal
    - Insurance riders
    - Network hospitals
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption(
        "Powered by Gemini, FAISS and Sentence Transformers."
    )

    st.caption("Insurance Policy Assistant | Version 1.0")


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📚 View Retrieved Policy Sources"):

                for source in message["sources"]:

                    st.markdown(f"**{source['title']}**")

                    st.caption(
                        f"Document ID: {source['doc_id']} | "
                        f"Category: {source['category']} | "
                        f"Source Type: {source['source_type']} | "
                        f"File: {source['source_file']}"
                    )

                    st.write(source["text"])
                    st.divider()


# --------------------------------
# CHAT INPUT
# --------------------------------

question = st.chat_input(
    "Ask your insurance policy question..."
)


# --------------------------------
# SINGLE RAG PIPELINE
# --------------------------------

if question:

    # Preserve previous conversation
    conversation_history = (
        st.session_state.messages.copy()
    )

    # Save current user question
    st.session_state.messages.append({

        "role": "user",
        "content": question

    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            # STEP 1: RETRIEVAL

            with st.spinner("Searching policy documents..."):

                start_time = time.perf_counter()

                results = retriever.search(
                    question,
                    top_k=10,
                    similarity_threshold=0.30
                )

                retrieval_time = (
                    time.perf_counter() - start_time
                )

            print(
                f"Retrieval time: {retrieval_time:.2f} seconds"
            )

            st.write("### Retrieval Debug Information")

            st.write("Selected source: JSON insurance policies")

            st.write("Retrieved documents:")

            for item in results:
                st.write(
                    item["doc_id"],
                    "|",
                    item["source_type"],
                    "|",
                    round(item["similarity_score"], 4)
                )
            
            # STEP 2: CHECK RETRIEVAL

            if not results:

                answer = (
                    "I couldn't find relevant information "
                    "in the available insurance policy documents. "
                    "Please try asking a more specific question."
                )

                st.warning(answer)

                st.session_state.messages.append({

                    "role": "assistant",
                    "content": answer,
                    "sources": []

                })

            else:

                # STEP 3: PREPARE CONTEXT

                start_time = time.perf_counter()

                
                context = "\n\n".join(
                    [
                        f"Document ID: {item['doc_id']}\n"
                        f"Title: {item['title']}\n"
                        f"Category: {item['category']}\n"
                        f"Source Type: {item['source_type']}\n"
                        f"Source File: {item['source_file']}\n"
                        f"Content: {item['text']}"
                        for item in results
                    ]
                )


                context_time = (
                    time.perf_counter() - start_time
                )

                print(
                    f"Context preparation time: "
                    f"{context_time:.2f} seconds"
                )

                # STEP 4: GENERATE ANSWER

                # STEP 4: GENERATE ANSWER

                if retrieval_only:

                    answer = (
                        "Retrieval successful! Gemini generation is skipped "
                        "because Retrieval Testing Mode is enabled."
                    )

                    generation_time = 0

                else:

                    with st.spinner("Generating answer using Gemini..."):

                        start_time = time.perf_counter()

                        answer = generator.generate_answer(
                            question,
                            context,
                            conversation_history
                        )

                        generation_time = (
                            time.perf_counter() - start_time
                        )

                    print(
                        f"Gemini generation time: "
                        f"{generation_time:.2f} seconds"
                    )

                # STEP 5: DISPLAY ANSWER

                st.markdown(answer)

                # STEP 6: DISPLAY SOURCES

                with st.expander("📚 View Retrieved Policy Sources"):

                    for source in results:

                        st.markdown(
                            f"**{source['title']}**"
                        )

                        st.caption(
                            f"Document ID: {source['doc_id']} | "
                            f"Category: {source['category']} | "
                            f"Source Type: {source['source_type']} | "
                            f"File: {source['source_file']}"
                        )

                        st.write(source["text"])
                        st.divider()

                # STEP 7: SAVE ASSISTANT RESPONSE

                st.session_state.messages.append({

                    "role": "assistant",
                    "content": answer,
                    "sources": results

                })

        except Exception as e:

            error_message = (
                "Something went wrong while processing "
                "your question. Please try again."
            )

            st.error(error_message)

            print(f"Application error: {e}")

            st.session_state.messages.append({

                "role": "assistant",
                "content": error_message,
                "sources": []

            })
