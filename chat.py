from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import pickle
import os

load_dotenv()

def load_base_retriever(base_retriever_path):
    try:
        with open(base_retriever_path, "rb") as f:
            bm25_retriever = pickle.load(f)
            bm25_retriever.k = 2
            return bm25_retriever
    except FileNotFoundError:
        print("Base retriever file not found.")
        return None
    except Exception as e:
        print(f"Error loading base retriever: {e}")
        return None

def load_standard_retriever(embedding_file_path):
    try:
        persist_directory = embedding_file_path
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return None

def load_retriever(user_data):
    try:
        base_retriever_path = os.path.join('media', 'semantic-search' ,user_data['username'], 'base_retriever_db',f'{user_data["file_name"]}_retriever.pkl')
        standard_retriever_path = os.path.join('media', 'semantic-search' ,user_data['username'], 'standard_db')
        hyde_retriever_path = os.path.join('media', 'semantic-search' ,user_data['username'], 'hyde_db')
        
        if user_data['retriever_type'] == 'ensemble':
            bm25_retriever = load_base_retriever(base_retriever_path)
            vector_store_retriever = load_standard_retriever(standard_retriever_path)
            
            if bm25_retriever and vector_store_retriever:
                retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_store_retriever], weights=[0.5, 0.5])
                
        elif user_data['retriever_type'] == 'standard':
            retriever = load_standard_retriever(standard_retriever_path)

        elif user_data['retriever_type'] == 'hyde':
            base_embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
            embeddings = HypotheticalDocumentEmbedder.from_llm(
                llm, base_embeddings, "web_search"
            )
            persist_directory = hyde_retriever_path
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

            retriever = vector_store.as_retriever(search_kwargs={"k":3})

        return retriever
    
    except Exception as e:
        print(f"Error in chat_with_doc: {e}")
        return None
    
async def search_relevant_documents(retriever, query, k=5):
    return await retriever.aget_relevant_documents(query=query, k=k)


async def run_chain(context, chat_memory, query):
    
    llm = ChatOpenAI(model_name="gpt-4" ,temperature=0)
    
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context} 
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm
    
    response = ""
        
    async for token in chain.astream({"context": context, "question": query}):  # type: ignore
        yield token.content
        response += token.content

    chat_memory.add_user_message(message=query)
    chat_memory.add_ai_message(message=response)
    

def process_llm_response(relevant_documents):
    sources = []
    for doc in relevant_documents:
        sources.append((doc.metadata['source'], doc.metadata['page']))
    return sources

def group_sources(sources):
    grouped_results = {}

    for source in sources:
        document_name = source[0].split('/')[-1]
        page_number = source[1]

        if document_name not in grouped_results:
            grouped_results[document_name] = [document_name, page_number]
        else:
            grouped_results[document_name].append(page_number)

    final_results = list(grouped_results.values())

    return final_results