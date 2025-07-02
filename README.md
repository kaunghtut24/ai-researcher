# Agentic Deep Researcher

This project features a multi-agent deep researcher powered by CrewAI, capable of performing deep web searches using Linkup. The application consists of a FastAPI backend for the agent orchestration and a Streamlit frontend for an interactive user interface.

We use:

- [LinkUp](https://www.linkup.so/) (Search Tool)
- CrewAI (Agentic design)
- OpenAI (LLM) - Configurable via `OPENAI_API_KEY`
- FastAPI for the backend API
- Streamlit for the interactive UI

### SetUp

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ai-researcher
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    Create a `.env` file in the project root or set the following environment variables:
    ```
    LINKUP_API_KEY=your_linkup_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    [Get your Linkup API keys here](https://www.linkup.so/)
    [Get your OpenAI API keys here](https://platform.openai.com/account/api-keys)

### Run the Application

To run the application, you need to start both the FastAPI backend and the Streamlit frontend.

```bash
# Start the FastAPI server in the background
python server.py &

# Start the Streamlit app
streamlit run app.py
```

## 📬 Stay Updated with Our Newsletter!

**Get a FREE Data Science eBook** 📖 with 150+ essential lessons in Data Science when you subscribe to our newsletter! Stay in the loop with the latest tutorials, insights, and exclusive resources. [Subscribe now!](https://join.dailydoseofds.com)

[![Daily Dose of Data Science Newsletter](https://github.com/patchy631/ai-engineering/blob/main/resources/join_ddods.png)](https://join.dailydoseofds.com)

## Contribution

Contributions are welcome! Feel free to fork this repository and submit pull requests with your improvements.