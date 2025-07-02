import os
from typing import Type
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from linkup import LinkupClient
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
import litellm

# Set logging level based on environment (DEBUG for development, INFO for production)
import os
log_level = os.getenv('LITELLM_LOG', 'INFO')
os.environ['LITELLM_LOG'] = log_level

# Load environment variables (for non-LinkUp settings)
load_dotenv()


def get_llm_client(llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None):
    """Initialize and return the LLM client based on the selected provider."""
    if llm_provider == "Ollama":
        # NOTE: Ollama base_url is set to localhost here. In a production environment
        # like Render, this will need to be an accessible URL for your Ollama instance.
        # If Ollama is running as a separate service, its URL should be provided here.
        return LLM(
            model=model,
            # base_url="http://localhost:11434" # Removed for now, user needs to configure if external
        )
    elif llm_provider == "OpenAI":
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required for OpenAI models.")
        return LLM(
            model=model,
            openai_api_key=openai_api_key
        )
    elif llm_provider == "OpenAI Compatible":
        if not openai_api_key:
            raise ValueError("API Key is required for OpenAI Compatible models.")
        if not openai_base_url:
            raise ValueError("Base URL is required for OpenAI Compatible models.")
        return LLM(
            model=f"openai/{model}",
            api_key=openai_api_key,
            base_url=openai_base_url
        )
    else:
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


def create_research_crew(query: str, llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None, document_content: str = ""):
    """Create and configure the research crew with all agents and tasks"""
    # Initialize tools
    linkup_search_tool = LinkUpSearchTool()

    # Get LLM client
    client = get_llm_client(llm_provider, model, openai_api_key, openai_base_url)

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


def run_research(query: str, llm_provider: str, model: str, openai_api_key: str = None, openai_base_url: str = None, document_content: str = ""):
    """Run the research process and return results"""
    print(f"Starting research for query: {query} with model: {model} using provider: {llm_provider}")
    try:
        crew = create_research_crew(query, llm_provider, model, openai_api_key, openai_base_url, document_content)
        print("Crew created successfully.")
        result = crew.kickoff()
        print("Crew kickoff completed.")
        return result.raw
    except Exception as e:
        print(f"An error occurred during research: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

