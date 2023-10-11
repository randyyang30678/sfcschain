from langchain.agents import AgentType, initialize_agent
from sfcschain.tools.load_tools import load_tools
from langchain.chat_models import ChatOpenAI

# Path: sfcschain/tools/sfcs_duty_search/tool.py

if __name__ == "main":
    tool_names = ["sfcs_duty_search", "sfcs_sitemap_search"]
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(tool_names, llm)

    agent = initialize_agent(
        tools=tools, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True
    )

    agent.run("How to resolve SFCS Duty '小板報廢'?")
