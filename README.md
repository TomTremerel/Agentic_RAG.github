# Agentic_RAG.github

Hi there, I finished my first Agentic_RAG using the powerful Crewai framework. 

Agentic_RAG is an improved RAG, because some task are shared by different agent and we can define some specific task to avoid the hallucination. 

![image](https://github.com/user-attachments/assets/020f44cf-f639-46b0-9aea-4a0370242d97)

**The workflow is the following** : 
The user send his query, the rooter agent analyze if the query is related to the document inside the vector db or not. Then the retriever agent is responsible of find the information and writting the good answer. Finally, the grader agent give a grade to the wrtting of the retriever agent and if it is relevent, this is the final output, if not the retriever agent rewrite its answer.  
