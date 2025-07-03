import os
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from linkup import LinkupClient
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import litellm
import logging

# Set logging level based on environment (DEBUG for development, INFO for production)
log_level = os.getenv('LITELLM_LOG', 'INFO')
os.environ['LITELLM_LOG'] = log_level

# Configure logging for better debugging
logging.basicConfig(level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)

# Load environment variables (for non-LinkUp settings)
load_dotenv()


def test_llm_connection(llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None):
    """Test LLM connection with a simple prompt using LiteLLM"""
    try:
        if llm_provider == "OpenAI Compatible":
            import litellm
            
            logger.info(f"Testing LLM connection for {model} at {openai_base_url}")
            
            # Set environment variables for LiteLLM
            import os
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["OPENAI_API_BASE"] = openai_base_url

            # Test using LiteLLM with the same configuration as CrewAI
            response = litellm.completion(
                model=model,  # Use model name directly
                messages=[{"role": "user", "content": "Hello, respond with 'OK' if you can hear me."}],
                max_tokens=10,
                temperature=0,
                api_key=openai_api_key,
                base_url=openai_base_url
            )

            if response.choices and response.choices[0].message.content:
                logger.info(f"LLM test successful: {response.choices[0].message.content}")
                return True
            else:
                logger.error("LLM test failed: Empty response")
                return False

    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        return False

def get_llm_client(llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None, ollama_base_url: str = None):
    """Initialize and return the LLM client based on the selected provider."""
    logger.info(f"Initializing LLM client for provider: {llm_provider}, model: {model}")
    
    if llm_provider == "Ollama":
        # Handle Ollama configuration
        base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(f"Configuring Ollama at {base_url}")

        try:
            return LLM(
                model=model,
                base_url=base_url
            )
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {base_url}: {e}")
            raise ValueError(f"Failed to connect to Ollama at {base_url}. Please ensure Ollama is running and accessible. Error: {str(e)}")

    elif llm_provider == "OpenAI":
        if not openai_api_key:
            logger.error("OpenAI API Key is required for OpenAI models.")
            raise ValueError("OpenAI API Key is required for OpenAI models.")
        logger.info("Configuring OpenAI LLM")
        return LLM(
            model=model,
            openai_api_key=openai_api_key
        )

    elif llm_provider == "OpenAI Compatible":
        if not openai_api_key:
            logger.error("API Key is required for OpenAI Compatible models.")
            raise ValueError("API Key is required for OpenAI Compatible models.")
        if not openai_base_url:
            logger.error("Base URL is required for OpenAI Compatible models.")
            raise ValueError("Base URL is required for OpenAI Compatible models.")

        logger.info(f"Configuring OpenAI Compatible LLM: {model} at {openai_base_url}")

        # Set up environment variables for LiteLLM
        import os

        # For LiteLLM with custom endpoints, use the model name directly
        # LiteLLM will automatically detect it's an OpenAI-compatible endpoint
        litellm_model = model

        logger.info(f"Using LiteLLM model format: {litellm_model}")
        logger.info(f"Base URL: {openai_base_url}")
        logger.debug(f"API Key: {'*' * (len(openai_api_key) - 4) + openai_api_key[-4:] if openai_api_key else 'None'}")

        # Set environment variables that LiteLLM expects
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["OPENAI_API_BASE"] = openai_base_url

        try:
            # Use the correct parameter names for LiteLLM
            logger.info("Attempting LLM configuration with base_url parameter")
            return LLM(
                model=litellm_model,
                api_key=openai_api_key,
                base_url=openai_base_url,
                temperature=0.1,
                max_tokens=4000
            )
        except Exception as e1:
            logger.warning(f"First attempt with base_url failed: {e1}")
            try:
                # Fallback using environment variables only
                logger.info("Attempting LLM configuration with environment variables only")
                return LLM(
                    model=litellm_model,
                    temperature=0.1,
                    max_tokens=4000
                )
            except Exception as e2:
                logger.warning(f"Second attempt with environment variables failed: {e2}")
                try:
                    # Final fallback using custom provider format
                    logger.info("Attempting LLM configuration with openai/ prefix")
                    return LLM(
                        model=f"openai/{model}",
                        api_key=openai_api_key,
                        base_url=openai_base_url,
                        temperature=0.1,
                        max_tokens=4000
                    )
                except Exception as e3:
                    logger.error(f"All attempts failed: {e1}, {e2}, {e3}")
                    raise ValueError(f"Failed to configure LLM for {model} at {openai_base_url}. Please check your configuration and try again.")

    else:
        logger.error(f"Unsupported LLM provider: {llm_provider}")
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

# Define LinkUp Search Tool


class LinkUpSearchInput(BaseModel):
    """Input schema for LinkUp Search Tool."""
    query: str = Field(description="The search query to perform")
    depth: str = Field(default="standard",
                       description="Depth of search: 'standard' or 'deep'")
    output_type: str = Field(
        default="searchResults", description="Output type: 'searchResults', 'sourcedAnswer', or 'structured'")


class LinkUpSearchTool(BaseTool):
    name: str = "LinkUp Search"
    description: str = "Search the web for information using LinkUp and return comprehensive results"
    args_schema: Type[BaseModel] = LinkUpSearchInput

    def __init__(self):
        super().__init__()

    def _run(self, query: str, depth: str = "standard", output_type: str = "searchResults") -> str:
        """Execute LinkUp search and return results."""
        try:
            # Initialize LinkUp client with API key from environment variables
            linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))

            # Perform search
            search_response = linkup_client.search(
                query=query,
                depth=depth,
                output_type=output_type
            )

            return str(search_response)
        except Exception as e:
            return f"Error occurred while searching: {str(e)}"


def create_research_crew(query: str, llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None, document_content: str = "", ollama_base_url: str = None):
    """Create and configure the research crew with all agents and tasks"""
    # Initialize tools
    linkup_search_tool = LinkUpSearchTool()

    # Get LLM client
    client = get_llm_client(llm_provider, model, openai_api_key, openai_base_url, ollama_base_url)

    web_searcher = Agent(
        role="Web Searcher",
        goal="Find the most relevant and comprehensive information on the web, along with source links (urls). Your search should be deep and wide, covering multiple perspectives and sources.",
        backstory="A master of the internet, capable of finding any information, no matter how obscure. You are a relentless researcher, always digging deeper to find the truth. You pass your findings to the 'Research Analyst'.",
        verbose=True,
        allow_delegation=False,
        tools=[linkup_search_tool],
        llm=client,
    )

    # Define the research analyst
    research_analyst = Agent(
        role="Research Analyst",
        goal="Analyze and synthesize raw information into structured, insightful, and comprehensive reports, along with source links (urls) as citations. Your analysis should be critical and well-supported by evidence.",
        backstory="A brilliant analyst, you can see the patterns that others miss. You are an expert at identifying key insights, verifying facts, and presenting complex information in a clear and concise manner. You can delegate fact-checking to the 'Web Searcher'. You pass your final analysis to the 'Technical Writer'.",
        verbose=True,
        allow_delegation=True,
        llm=client,
    )

    # Define the technical writer
    technical_writer = Agent(
        role="Technical Writer",
        goal="Create well-structured, clear, lengthy, and comprehensive responses in markdown format, with citations/source links (urls). Your writing should be engaging and informative, providing a deep dive into the topic.",
        backstory="A master of words, you can make any topic interesting and understandable. You are an expert at crafting compelling narratives and presenting information in a way that is both accessible and authoritative.",
        verbose=True,
        allow_delegation=False,
        llm=client,
    )

    # Define tasks
    search_task = Task(
        description=f"Search for comprehensive and in-depth information about: {query}.",
        agent=web_searcher,
        expected_output="A detailed report of raw search results, including a wide range of sources with URLs, formatted as a plain string.",
        tools=[linkup_search_tool]
    )

    analysis_task_description = "Analyze the raw search results, identify key information, verify facts, and prepare a structured and comprehensive analysis. The analysis should be well-supported by evidence and include multiple perspectives."
    if document_content:
        analysis_task_description += f"\n\nAdditional context from uploaded document:\n\n```\n{document_content}\n```"

    analysis_task = Task(
        description=analysis_task_description,
        agent=research_analyst,
        expected_output="A comprehensive and insightful analysis of the information, with verified facts, key insights, and source links, formatted as a plain string.",
        context=[search_task]
    )

    writing_task = Task(
        description="Create a lengthy, comprehensive, and well-organized response based on the research analysis. The response should be in markdown format and include proper citations/source links (urls).",
        agent=technical_writer,
        expected_output="A clear, lengthy, and comprehensive response that directly answers the query with proper citations/source links (urls). The response should be well-structured, engaging, and provide a deep dive into the topic.",
        context=[analysis_task]
    )

    # Create the crew
    crew = Crew(
        agents=[web_searcher, research_analyst, technical_writer],
        tasks=[search_task, analysis_task, writing_task],
        verbose=True,
        process=Process.sequential
    )

    return crew


def run_research(query: str, llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None, document_content: str = "", ollama_base_url: str = None):
    """Run the research process and return results"""
    logger.info(f"Starting research for query: {query}")
    logger.info(f"LLM Provider: {llm_provider}")
    logger.info(f"Model: {model}")
    logger.info(f"Base URL: {openai_base_url}")

    try:
        # Test LLM connection first for OpenAI Compatible providers
        if llm_provider == "OpenAI Compatible":
            logger.info("Testing LLM connection...")
            if not test_llm_connection(llm_provider, model, openai_api_key, openai_base_url):
                error_msg = f"LLM Connection Failed: Cannot connect to {model} at {openai_base_url}. Please verify:\n1. API key is correct\n2. Base URL is correct\n3. Model name is exact (case-sensitive)\n4. You have access to this model"
                logger.error(error_msg)
                return error_msg

        # Test LLM configuration
        logger.info("Creating LLM client...")
        client = get_llm_client(llm_provider, model, openai_api_key, openai_base_url, ollama_base_url)
        logger.info("LLM client created successfully.")

        # Create crew
        logger.info("Creating research crew...")
        crew = create_research_crew(query, llm_provider, model, openai_api_key, openai_base_url, document_content, ollama_base_url)
        logger.info("Crew created successfully.")

        # Execute research
        logger.info("Starting crew execution...")
        result = crew.kickoff()
        logger.info("Crew execution completed.")

        if result and hasattr(result, 'raw') and result.raw:
            return result.raw
        else:
            logger.warning("Research completed but no results were returned. This might be due to LLM configuration issues.")
            return "Research completed but no results were returned. This might be due to LLM configuration issues."

    except Exception as e:
        error_msg = str(e)
        logger.error(f"An error occurred during research: {error_msg}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        # Provide more specific error messages
        if "Invalid response from LLM call" in error_msg or "None or empty" in error_msg:
            specific_error = f"LLM Configuration Error: The model '{model}' at '{openai_base_url}' is not responding properly. Please check:\n1. API key is valid\n2. Base URL is correct\n3. Model name is exact\n4. API provider supports the model\n\nOriginal error: {error_msg}"
            logger.error(specific_error)
            return specific_error
        else:
            general_error = f"Research Error: {error_msg}"
            logger.error(general_error)
            return general_error

