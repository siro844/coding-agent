import os
from dotenv import load_dotenv
from operator import itemgetter
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model='claude-3-opus-20240229',streaming=True)
llm =GoogleGenerativeAI(model="gemini-pro",streaming=True)
url="https://python.langchain.com/docs/expression_language/"
loader=RecursiveUrlLoader(
    url=url,
    max_depth=20,
    extractor=lambda x:Soup(x,"html.parser").text
)
docs=loader.load()

#Sort the list based on URLS in 'metadata' ->'Source'
docs_sorted=sorted(docs,key=lambda x:x.metadata["source"])
docs_reversed=list(reversed(docs_sorted))

#Concatenate page_content of each sorted dictionary
concatenated_content="\n\n\n----\n\n\n".join([doc.page_content for doc in docs_reversed])

# with open('docs.txt', 'w', encoding='utf-8') as f:
#     f.write(concatenated_content)
class code(BaseModel):
    """ Code Output """
    prefix:str=Field(description="Description of the problem and approach")
    imports:str=Field(description="Code block import Statements")
    code:str=Field(description="Code block not including import statements")


parser=PydanticOutputParser(pydantic_object=code)
template="""
You are a coding assistant with expertise in LCEL,langchain expresssion language.\n
Here is a full set of LCEL documentaion:
\n---------\n
{context}
\n---------\n
Answer the questions based on above documentaion.\n
Ensure the code you provided can be executed with all required imports and variables and imports defined.\n
Structure your answer with description of code solution.\n
Then list the imports then finally the functioning code block.\n
Here is the user question :\n
--- --- --- --- \n
{question}
"""
prompt=PromptTemplate(
    template=template,
    input_variables=["context","question"],
)
chain=(
    {
        "context":lambda x:concatenated_content,
        "question":itemgetter("question"),
    }|prompt|llm|parser
)

print(type(chain.invoke({
    "question":"How to create a agent in LCEL?"
})))
