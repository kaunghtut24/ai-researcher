import streamlit as st
from agents import run_research
import os
import weasyprint
from markdown_it import MarkdownIt
import pyperclip
import json
import requests
import io

# Set up page configuration
st.set_page_config(page_title="🔍 Agentic Deep Researcher", layout="wide")

# --- Persistence --- #

USER_DATA_FILE = "user_data.json"

def save_user_data():
    data = {
        "linkup_api_key": st.session_state.linkup_api_key,
        "selected_model": st.session_state.selected_model,
        "selected_llm_provider": st.session_state.selected_llm_provider,
        "openai_api_key": st.session_state.openai_api_key,
        "openai_base_url": st.session_state.openai_base_url,
        "messages": st.session_state.messages,
        "uploaded_document_content": st.session_state.uploaded_document_content
    }
    with open(USER_DATA_FILE, "w") as f:
        json.dump(data, f)

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

# --- Model Selection --- #

# NOTE: Ollama is assumed to be a separate service.
# If running on Render, this will likely fail unless Ollama is also deployed and accessible.
@st.cache_data
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json()["models"]]
    except Exception as e:
        st.warning(f"Could not connect to Ollama at http://localhost:11434. Please ensure Ollama is running if you intend to use it. Error: {e}")
        return [] # Return empty list if Ollama is not available

# --- App --- #

# Initialize session state variables
user_data = load_user_data()
st.session_state.linkup_api_key = user_data.get("linkup_api_key", "")
st.session_state.selected_model = user_data.get("selected_model", "gpt-4o")
st.session_state.selected_llm_provider = user_data.get("selected_llm_provider", "OpenAI")
st.session_state.openai_api_key = user_data.get("openai_api_key", "")
st.session_state.openai_base_url = user_data.get("openai_base_url", "https://api.openai.com/v1")
st.session_state.messages = user_data.get("messages", [])
st.session_state.uploaded_document_content = user_data.get("uploaded_document_content", "")

def reset_chat():
    st.session_state.messages = []
    st.session_state.uploaded_document_content = ""
    save_user_data()

def to_pdf(html_string):
    return weasyprint.HTML(string=html_string).write_pdf()

# Sidebar: Linkup Configuration with updated logo link
with st.sidebar:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("")
        st.image(
            "https://avatars.githubusercontent.com/u/175112039?s=200&v=4", width=65)
    with col2:
        st.header("Linkup Configuration")
        st.write("Deep Web Search")

    st.markdown("[Get your API key](https://app.linkup.so/sign-up)",
                unsafe_allow_html=True)

    linkup_api_key = st.text_input(
        "Enter your Linkup API Key", type="password", value=st.session_state.linkup_api_key)
    if linkup_api_key:
        st.session_state.linkup_api_key = linkup_api_key
        os.environ["LINKUP_API_KEY"] = linkup_api_key
        st.success("API Key stored successfully!")
        save_user_data()

    st.header("LLM Provider")
    llm_providers = ["Ollama", "OpenAI", "OpenAI Compatible"]
    st.session_state.selected_llm_provider = st.selectbox("Select LLM Provider", llm_providers, index=llm_providers.index(st.session_state.selected_llm_provider))
    save_user_data()

    if st.session_state.selected_llm_provider == "Ollama":
        st.header("Ollama Model Selection")
        models = get_ollama_models()
        selected_model_raw = st.selectbox("Select a model", models, index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0)
        if not selected_model_raw.startswith("ollama/"):
            st.session_state.selected_model = f"ollama/{selected_model_raw}"
        else:
            st.session_state.selected_model = selected_model_raw
        save_user_data()
    elif st.session_state.selected_llm_provider == "OpenAI":
        st.header("OpenAI Configuration")
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key", type="password", value=st.session_state.openai_api_key)
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("OpenAI API Key stored successfully!")
            save_user_data()
        st.session_state.selected_model = st.text_input("Enter OpenAI Model Name", value=st.session_state.selected_model if st.session_state.selected_model else "gpt-4o")
        save_user_data()
    elif st.session_state.selected_llm_provider == "OpenAI Compatible":
        st.header("OpenAI Compatible Configuration")
        openai_api_key = st.text_input(
            "Enter your API Key", type="password", value=st.session_state.openai_api_key)
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("API Key stored successfully!")
            save_user_data()
        openai_base_url = st.text_input("Enter Base URL", value=st.session_state.openai_base_url)
        if openai_base_url:
            st.session_state.openai_base_url = openai_base_url
            save_user_data()
        st.session_state.selected_model = st.text_input("Enter Model Name", value=st.session_state.selected_model if st.session_state.selected_model else "gpt-4o")
        save_user_data()

    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a document (TXT, MD)", type=["txt", "md"])
    if uploaded_file is not None:
        string_io = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.session_state.uploaded_document_content = string_io.read()
        st.success("Document uploaded successfully!")
        save_user_data()
    elif st.session_state.uploaded_document_content:
        st.info("Document already loaded. Upload a new one to replace.")

    use_document = st.checkbox("Use uploaded document", value=True)

# Main Chat Interface Header with powered by logos from original code links
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("<h2 style='color: #0066cc;'>🔍 Agentic Deep Researcher</h2>",
                unsafe_allow_html=True)
    powered_by_html = """
    <div style='display: flex; align-items: center; gap: 10px; margin-top: 5px;'>
        <span style='font-size: 20px; color: #666;'>Powered by</span>
        <img src="https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg" width="80"> 
        <span style='font-size: 20px; color: #666;'>and</span>
        <img src="https://framerusercontent.com/images/wLLGrlJoyqYr9WvgZwzlw91A8U.png?scale-down-to=512" width="100">
    </div>
    """
    st.markdown(powered_by_html, unsafe_allow_html=True)
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Add spacing between header and chat history
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Save as PDF",
                    data=to_pdf(MarkdownIt().render(message["content"])),
                    file_name=f"research_report_{i}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{i}"
                )
            with col2:
                if st.button("Copy", key=f"copy_{i}"):
                    pyperclip.copy(message["content"])
                    st.success("Copied to clipboard!")


# Accept user input and process the research query
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.linkup_api_key:
        response = "Please enter your Linkup API Key in the sidebar."
    else:
        with st.spinner("Researching... This may take a moment..."):
            try:
                result = run_research(
                    prompt,
                    st.session_state.selected_llm_provider,
                    st.session_state.selected_model,
                    st.session_state.openai_api_key,
                    st.session_state.openai_base_url,
                    st.session_state.uploaded_document_content if use_document else ""
                )
                response = result
            except Exception as e:
                response = f"An error occurred: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(response)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Save as PDF",
                data=to_pdf(MarkdownIt().render(response)),
                file_name="research_report_latest.pdf",
                mime="application/pdf",
                key="download_pdf_latest"
            )
        with col2:
            if st.button("Copy", key="copy_latest"):
                pyperclip.copy(response)
                st.success("Copied to clipboard!")

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
    save_user_data()

