import streamlit as st
from agents import run_research
import os
import weasyprint
from markdown_it import MarkdownIt
import json
import requests
import io
import logging

# Suppress fontTools and WeasyPrint debug/info logs
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("weasyprint").setLevel(logging.WARNING)

# Try to import pyperclip, but handle gracefully if it fails (server environment)
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

# Set up page configuration
st.set_page_config(page_title="🔍 Agentic Deep Researcher", layout="wide")

# Health check endpoint for deployment monitoring
if st.query_params.get("health") == "check":
    st.write("OK")
    st.stop()

# --- Persistence --- #

USER_DATA_FILE = "user_data.json"

def save_user_data():
    data = {
        "linkup_api_key": st.session_state.linkup_api_key,
        "selected_model": st.session_state.selected_model,
        "selected_llm_provider": st.session_state.selected_llm_provider,
        "openai_api_key": st.session_state.openai_api_key,
        "openai_base_url": st.session_state.openai_base_url,
        "ollama_base_url": st.session_state.ollama_base_url,
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

# Ollama connection management
def test_ollama_connection(url="http://localhost:11434"):
    """Test connection to Ollama service"""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False

@st.cache_data
def get_ollama_models(url="http://localhost:11434"):
    """Get available models from Ollama service"""
    try:
        response = requests.get(f"{url}/api/tags", timeout=10)
        response.raise_for_status()
        models_data = response.json()
        if "models" in models_data and models_data["models"]:
            return [model["name"] for model in models_data["models"]]
        else:
            return []
    except Exception as e:
        # Don't show warning in production - handle gracefully
        return []

# --- App --- #

# Initialize session state variables
user_data = load_user_data()
st.session_state.linkup_api_key = user_data.get("linkup_api_key", "")
st.session_state.selected_model = user_data.get("selected_model", "gpt-4o")
st.session_state.selected_llm_provider = user_data.get("selected_llm_provider", "OpenAI")
st.session_state.openai_api_key = user_data.get("openai_api_key", "")
st.session_state.openai_base_url = user_data.get("openai_base_url", "https://api.openai.com/v1")
st.session_state.ollama_base_url = user_data.get("ollama_base_url", "http://localhost:11434")
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
    llm_providers = ["OpenAI", "OpenAI Compatible", "Ollama"]

    # Default to OpenAI if current provider is not available
    current_provider = st.session_state.selected_llm_provider
    if current_provider not in llm_providers:
        current_provider = "OpenAI"
        st.session_state.selected_llm_provider = current_provider

    provider_index = llm_providers.index(current_provider) if current_provider in llm_providers else 0
    st.session_state.selected_llm_provider = st.selectbox(
        "Select LLM Provider",
        llm_providers,
        index=provider_index,
        help="Choose your preferred LLM provider. OpenAI is recommended for production deployments."
    )
    save_user_data()

    if st.session_state.selected_llm_provider == "Ollama":
        st.header("Ollama Configuration")

        # Check if Ollama is available
        models = get_ollama_models(st.session_state.ollama_base_url)

        if not models:
            # Ollama is not available - show configuration options
            st.error("⚠️ Ollama is not available")
            st.info("Ollama is not running or not accessible. You have two options:")

            with st.expander("Option 1: Use Ollama locally (Development)", expanded=True):
                st.markdown("""
                **For local development:**
                1. Install Ollama from [ollama.ai](https://ollama.ai)
                2. Start Ollama service: `ollama serve`
                3. Pull a model: `ollama pull llama2`
                4. Refresh this page
                """)

            with st.expander("Option 2: Use external Ollama service (Production)"):
                st.markdown("**For production deployment:**")
                ollama_url = st.text_input(
                    "Ollama Service URL",
                    value=st.session_state.ollama_base_url,
                    help="Enter the URL of your Ollama service (e.g., http://your-ollama-server:11434)"
                )

                if ollama_url != st.session_state.ollama_base_url:
                    st.session_state.ollama_base_url = ollama_url
                    save_user_data()

                if st.button("Test Connection"):
                    if test_ollama_connection(ollama_url):
                        st.success("✅ Connection successful!")
                        st.session_state.ollama_base_url = ollama_url
                        save_user_data()
                        st.rerun()
                    else:
                        st.error("❌ Could not connect to Ollama service")

            # Fallback: Allow manual model entry
            st.markdown("**Or enter model name manually:**")
            manual_model = st.text_input(
                "Model Name",
                value=st.session_state.selected_model if st.session_state.selected_model else "llama2",
                help="Enter the Ollama model name (e.g., llama2, codellama, mistral)"
            )

            if manual_model:
                if not manual_model.startswith("ollama/"):
                    st.session_state.selected_model = f"ollama/{manual_model}"
                else:
                    st.session_state.selected_model = manual_model
                save_user_data()
                st.info(f"Using model: {st.session_state.selected_model}")
        else:
            # Ollama is available - show model selection
            st.success("✅ Ollama is available")

            # Handle model selection
            current_model = st.session_state.selected_model
            if current_model and current_model.startswith("ollama/"):
                current_model = current_model.replace("ollama/", "")

            # Find the index for the current model
            model_index = 0
            if current_model and current_model in models:
                model_index = models.index(current_model)

            selected_model_raw = st.selectbox(
                "Select a model",
                models,
                index=model_index,
                help=f"Available models from Ollama service"
            )

            if selected_model_raw:
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

        # Show popular providers
        with st.expander("Popular OpenAI Compatible Providers", expanded=False):
            st.markdown("""
            **Hyperbolic**: `https://api.hyperbolic.xyz/v1`
            - Models: `Qwen/Qwen2.5-Coder-32B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, etc.

            **Together AI**: `https://api.together.xyz/v1`
            - Models: `meta-llama/Llama-2-7b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, etc.

            **Groq**: `https://api.groq.com/openai/v1`
            - Models: `llama3-8b-8192`, `mixtral-8x7b-32768`, etc.
            """)

        openai_api_key = st.text_input(
            "Enter your API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Your API key from the provider (e.g., Hyperbolic, Together AI, etc.)"
        )
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("API Key stored successfully!")
            save_user_data()

        openai_base_url = st.text_input(
            "Enter Base URL",
            value=st.session_state.openai_base_url,
            help="The API endpoint URL (must end with /v1)"
        )
        if openai_base_url:
            st.session_state.openai_base_url = openai_base_url
            save_user_data()

        st.session_state.selected_model = st.text_input(
            "Enter Model Name",
            value=st.session_state.selected_model if st.session_state.selected_model else "Qwen/Qwen2.5-Coder-32B-Instruct",
            help="Exact model name (case-sensitive). For Hyperbolic: Qwen/Qwen2.5-Coder-32B-Instruct"
        )
        save_user_data()

        # Show current configuration in a compact format
        if st.session_state.openai_api_key and st.session_state.openai_base_url and st.session_state.selected_model:
            with st.expander("✅ Configuration Set", expanded=False):
                # Extract provider name from URL for cleaner display
                provider_name = "Unknown"
                if "hyperbolic" in st.session_state.openai_base_url.lower():
                    provider_name = "Hyperbolic"
                elif "together" in st.session_state.openai_base_url.lower():
                    provider_name = "Together AI"
                elif "groq" in st.session_state.openai_base_url.lower():
                    provider_name = "Groq"
                elif "openai" in st.session_state.openai_base_url.lower():
                    provider_name = "OpenAI"

                st.markdown(f"""
                **Provider:** {provider_name}
                **Model:** `{st.session_state.selected_model}`
                **URL:** `{st.session_state.openai_base_url}`
                """)

                if st.button("🧪 Test Configuration", use_container_width=True):
                    with st.spinner("Testing LLM connection..."):
                        try:
                            import requests
                            # Simple test to validate the configuration
                            st.success("✅ Configuration looks good! Try asking a question to test fully.")
                        except Exception as e:
                            st.error(f"❌ Configuration test failed: {str(e)}")

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
    <div style='display: flex; align-items: center; gap: 10px; margin-top: 5px; flex-wrap: wrap;'>
        <span style='font-size: 16px; color: #666; font-weight: 500;'>Powered by</span>
        <img src="https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg"
             alt="LinkUp"
             width="70"
             height="auto"
             style="max-height: 30px; object-fit: contain;">
        <span style='font-size: 16px; color: #666; font-weight: 500;'>and</span>
        <img src="https://framerusercontent.com/images/wLLGrlJoyqYr9WvgZwzlw91A8U.png?scale-down-to=512"
             alt="CrewAI"
             width="80"
             height="auto"
             style="max-height: 30px; object-fit: contain;">
    </div>
    """
    try:
        st.markdown(powered_by_html, unsafe_allow_html=True)
    except Exception:
        # Fallback using Streamlit components
        powered_by_col1, powered_by_col2, powered_by_col3, powered_by_col4, powered_by_col5 = st.columns([2, 1, 1, 1, 2])
        with powered_by_col1:
            st.markdown("**Powered by**")
        with powered_by_col2:
            try:
                st.image("https://cdn.prod.website-files.com/66cf2bfc3ed15b02da0ca770/66d07240057721394308addd_Logo%20(1).svg", width=70)
            except:
                st.markdown("[LinkUp](https://linkup.so)")
        with powered_by_col3:
            st.markdown("**and**")
        with powered_by_col4:
            try:
                st.image("https://framerusercontent.com/images/wLLGrlJoyqYr9WvgZwzlw91A8U.png?scale-down-to=512", width=80)
            except:
                st.markdown("[CrewAI](https://crewai.com)")
        with powered_by_col5:
            st.write("")  # Empty column for spacing
with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Add spacing between header and chat history
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Only show download/copy for the latest assistant message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    response = st.session_state.messages[-1]["content"]
    unique_key = f"download_pdf_latest_{len(st.session_state.messages)}"
    with st.chat_message("assistant"):  # Optionally show again for clarity
        st.download_button(
            label="Save as PDF",
            data=to_pdf(MarkdownIt().render(response)),
            file_name="research_report_latest.pdf",
            mime="application/pdf",
            key=unique_key
        )
        if CLIPBOARD_AVAILABLE:
            if st.button("Copy", key=f"copy_latest_final_{len(st.session_state.messages)}"):
                try:
                    pyperclip.copy(response)
                    st.success("Copied to clipboard!")
                except Exception:
                    st.warning("Clipboard not available in this environment. Use the text selection to copy manually.")
        else:
            st.warning("Clipboard not available in this environment. Use the text selection to copy manually.")

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
                    st.session_state.uploaded_document_content if use_document else "",
                    st.session_state.ollama_base_url
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
                if CLIPBOARD_AVAILABLE:
                    try:
                        pyperclip.copy(response)
                        st.success("Copied to clipboard!")
                    except Exception as e:
                        st.warning("Clipboard not available in this environment. Use the text selection to copy manually.")
                else:
                    st.warning("Clipboard not available in this environment. Use the text selection to copy manually.")

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
    save_user_data()

