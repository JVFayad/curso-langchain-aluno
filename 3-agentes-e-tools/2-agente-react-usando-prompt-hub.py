from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result as a string."""
    try:
        result = eval(expression)  # cuidado: apenas para exemplo didático
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool
def web_search_mock(query: str) -> str:
    """Return the capital of a given country if it exists in the mock data."""
    data = {
        "Brazil": "Brasília",
        "France": "Paris",
        "Germany": "Berlin",
        "Italy": "Rome",
        "Spain": "Madrid",
        "United States": "Washington, D.C."
        
    }
    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "I don't know the capital of that country."


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
tools = [calculator, web_search_mock]

# Create agent with the new LangChain 1.x API
system_prompt = hub.pull("hwchase17/react")
agent = create_agent(llm, tools, system_prompt=system_prompt)

# The agent returns a graph that can be invoked directly
print(agent.invoke({"messages": [("user", "What is the capital of Brazil?")]}))
# print(agent.invoke({"messages": [("user", "How much is 10 + 10?")]}))