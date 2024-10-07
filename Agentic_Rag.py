# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:41:38 2024

@author: TOM
"""

import os
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool
from tavily import TavilyClient
from crewai_tools import tool
from crewai import Agent
from crewai import Crew
from crewai import Task


llm = ChatOpenAI(
    base_url = "https://api.groq.com/openai/v1",
    openai_api_key = 'GROQ_API_KEY',
    model_name = "llama-3.1-8b-instant",
    temperature=1,
    max_tokens=1024
    )

rag = PDFSearchTool(pdf ="NIPS-2017-attention-is-all-you-need-Paper.pdf",
                    config = dict(
                        llm=dict(
                            provider = "groq",
                            config=dict(
                                api_key = 'PDFST_API_KEY',
                                model ="llama-3.1-8b-instant"),
                            ),
                        embedder = dict(
                            provider="huggingface",
                            config=dict(
                                model="BAAI/bge-small-en"),
                            ),
                        )                 
    )


web_tool_search = TavilyClient(api_key='WEBT_API_KEY')

@tool
def router_tool(question):
    """Rooter Function"""
    if 'attention' or 'transformer' in question:
        return 'vectorstore'
    else:
          return 'web_search'
     
Router_Agent = Agent(
    role='rooter',
    goal='route user question to the vectorstore or the web search',
    backstory = (
    "you are a router expert agent at routing user question to vectorstore or web search"
    "Use the vectorstore for search related to transformers and attention mechanism"
    ),
    verbose = True,
    allow_delegation = False, 
    llm=llm)

Retriever_Agent = Agent(
    role= "retriever",
    goal= "use the information in the vectorstore to answer to the question",
    backstory=(
    "You are an expert question answering based on the vectorstore"
    "Use the information in the retrived context to write your answer to the question"
    "Be clear and precise in the answer"),
    verbose= True,
    allow_delegation= False,
    llm = llm)

Grader_Agent = Agent(
    role ="Answer grader",
    goal = "make sure the answer is relevent based on the question",
    backstory = (
    "You are a grader expert, you have to make sure the answer is relevent based on the question and the retrived document"
    "If the answer contain the keywords of the question, grade it as relevent"
    "If it is not relevent rewrite the answer"), 
    verbose = True,
    allow_delegation= False,
    llm=llm)
    

router_task = Task(
    description=("Analyse the keywords in the question {question}"
    "Based on the keywords decide whether it is eligible for a vectorstore search or a web search."
    "Return a single word 'vectorstore' if it is eligible for vectorstore search."
    "Return a single word 'websearch' if it is eligible for web search."
    "Do not provide any other premable or explaination."
    ),
    expected_output=("Give a binary choice 'websearch' or 'vectorstore' based on the question"
    "Do not provide any other premable or explaination."),
    agent=Router_Agent,
    tools=[router_tool],
)    

retriever_task = Task(
    description=("Based on the response from the router task extract information for the question {question} with the help of the respective tool."
    "Use the web_serach_tool to retrieve information from the web in case the router task output is 'websearch'."
    "Use the rag_tool to retrieve information from the vectorstore in case the router task output is 'vectorstore'."
    ),
    expected_output=("You should analyse the output of the 'router_task'"
    "If the response is 'websearch' then use the web_search_tool to retrieve information from the web."
    "If the response is 'vectorstore' then use the rag_tool to retrieve information from the vectorstore."
    "Return a claer and consise text as response."),
    agent=Retriever_Agent,
    context=[router_task],
)

grader_task = Task(
    description=("Based on the response from the retriever task for the quetion {question} evaluate whether the retrieved content is relevant to the question."
    ),
    expected_output=("Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question"
    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked."
    "You must answer 'no' if the response from the 'retriever_task' is not in alignment with the question asked."
    "Do not provide any preamble or explanations except for 'yes' or 'no'."),
    agent=Grader_Agent,
    context=[retriever_task],
)

rag_crew = Crew(
    agents=[Router_Agent, Retriever_Agent, Grader_Agent],
    tasks=[router_task, retriever_task, grader_task],
    verbose=True,

)

inputs = {"question" : "what is a transformer?"}
result = rag_crew.kickoff(inputs=inputs)   
