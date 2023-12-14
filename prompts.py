import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone



LETTER_TEMPLATE = """ You are tasked with retrieving questions regarding K-10 annual report by Guidewire with ticker name GWRE. Guidewire and GWRE must be used interchangeably by the user in question.

Provide an answer based on this retreival, and if you can't find anything relevant, just say "I'm sorry, I couldn't find that."

{context}

Question: {question}
Answer:
"""
LETTER_PROMPT = PromptTemplate(input_variables=["question", "context"], template=LETTER_TEMPLATE, )

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1000, 
    openai_api_key=st.secrets["openai_key"]
)


def get_faiss():
    " get the loaded FAISS embeddings"
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    return FAISS.load_local("faiss_index", embeddings)




def fs_chain(question):
    """
    returns a question answer chain for faiss vectordb
    """

    docsearch = get_faiss()
    qa_chain = RetrievalQA.from_chain_type(llm, 
                                           retriever=docsearch.as_retriever(),
                                           chain_type_kwargs={"prompt": FS_PROMPT})
    return qa_chain({"query": question})


