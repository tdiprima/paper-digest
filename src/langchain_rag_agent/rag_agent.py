import os
from datetime import datetime, timezone

import requests
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def fetch_weather_forecast(city, api_key):
    """
    Fetch Weather Data from OpenWeatherMap
    Get the 5-day forecast for a city
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&units=metric&appid={api_key}"
    resp = requests.get(url, timeout=10)
    data = resp.json()
    return data["list"]


def store_forecasts_in_faiss(forecasts, city):
    """
    Store Forecasts in FAISS Vector Store
    Each forecast entry is treated as a document. We'll use LangChain's FAISS wrapper.
    """
    docs = []
    for forecast in forecasts:
        # dt = datetime.utcfromtimestamp(forecast['dt']).strftime('%Y-%m-%d %H:%M')  # Deprecated in Python 3.12
        dt = datetime.fromtimestamp(forecast["dt"], tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M"
        )
        temp = forecast["main"]["temp"]
        weather = forecast["weather"][0]["description"]
        doc = Document(
            page_content=f"Date: {dt}, City: {city}, Temp: {temp}°C, Weather: {weather}",
            metadata={"date": dt, "city": city},
        )
        docs.append(doc)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    return vectordb


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_agent(vectordb):
    """
    Create a RetrievalQA Chain with Memory
    Using the new LangChain approach with RunnableWithMessageHistory.
    """
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
        MessagesPlaceholder("chat_history"),
        ("human", "Context: {context}\n\nQuestion: {input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain


if __name__ == "__main__":
    city = "London, UK"
    forecasts = fetch_weather_forecast(city, OPENWEATHER_API_KEY)
    vectordb = store_forecasts_in_faiss(forecasts, city)
    agent = build_agent(vectordb)

    session_id = "default_session"
    
    # Example multi-step query
    query = "What's the forecast trend for next week? Is it getting warmer or colder?"
    result = agent.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )
    print(result)

    # Follow-up question
    followup = "What about the chance of rain?"
    result2 = agent.invoke(
        {"input": followup},
        config={"configurable": {"session_id": session_id}}
    )
    print(result2)
