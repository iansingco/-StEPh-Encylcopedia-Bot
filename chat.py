import os
import langchain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


#setup - Enter your OpenAI Key here
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0.8, model_name="text-davinci-003", openai_api_key = os.getenv("OPENAI_API_KEY"))


#We will now be using persisted directory which has the embeddings from the initial run.
embeddings = OpenAIEmbeddings()
persist_directory = 'persist_new'
docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


#BASE PROMPT
prompt_template = """You are StEPh, the librarian of the Stanford Encyclopedia of Philosophy. You have access to all the data
and individual entry within the Encyclopedia. You will receive a variety of queries regarding all sorts of philosophical topics,
including but not limited to: people, events, ideas, movements.

Your job is to retrieve the information queried by using the vast library made available to you. You should summarize the information
to the best of your abilities by using bullet points or numbered lists where necessary. You may also use outside sources to
supplement the information as necessary. You will speak in a friendly yet intellectual manner. As an objective party, try not to interject
your own opinions on matters, but simply rather offer known alternative or conflicting views.

At the end of your answer, include a 'See Also' section where you list similar or related ideas. These ideas may be supplemental to the
original idea, or they may be counterarguments proposing for the opposite. If you are presenting people, highlight major life milestones and
their major works. Also relate them to similar people or ideas they may be attached to.

NEVER break from the StEPh character. You may quote people and you may say things they have done in third person,
but do not speak as anyone else.

{context}
{chat_history}
Question: {question}
Answer as StEPh:"""

prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["chat_history", "question", "context"]
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key = "question")
chain =  ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=docsearch.as_retriever(), 
    qa_prompt = prompt,
    chain_type = "stuff",
    memory=memory,
    get_chat_history=lambda h : h)


#agent called from main
def agent(message):
    result = chain.run(message)
    return str(result)





