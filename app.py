from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from langchain.memory import FileChatMessageHistory
import chat

app = FastAPI()
user_data_dict = {}

@app.post("/chat/sse/")
async def chat_sse():
    
    question = "Where is the parking level structure located?"
    file_name = "Scopesplit_1"
    username = "Hasnain"
    id = None

    retriever_type = "standard"

    user_data = {
        
        'username': username,
        'id': id,
        'retriever_type': retriever_type,
        'file_name': file_name
        }
    
    memory_json = f"./media/semantic-search/{username}/session.json"
    chat_memory = FileChatMessageHistory(file_path=memory_json)
    
    if username not in user_data_dict or user_data_dict[username]['retriever_type'] != retriever_type:
        
        user_data_dict[username] = {   
            'retriever' : chat.load_retriever(user_data),
            'retriever_type' : retriever_type}
    
    relevant_documents = await chat.search_relevant_documents(user_data_dict[username]['retriever'],
                                                              query=question)
    sources = chat.process_llm_response(relevant_documents)
    group_result = chat.group_sources(sources)
    
    return StreamingResponse(chat.run_chain(
        context=relevant_documents,
        chat_memory=chat_memory,
        query=question),
        media_type="text/event-stream")
    
#uvicorn app:app --reload