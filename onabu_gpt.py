from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant 
from langchain.embeddings import HuggingFaceEmbeddings
import qdrant_client
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from streamlit_chat import message
import pandas as pd
import streamlit_ext as ste

os.environ["QDRANT_HOST"]=st.secrets["QDRANT_HOST"]
os.environ["QDRANT_API_KEY"]=st.secrets["QDRANT_API_KEY"]
os.environ["QDRANT_COLLECTION_NAME"]=st.secrets["QDRANT_COLLECTION_NAME"]
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]



system_message = "You are an experienced agile coach that gives insightful information to customer within 300 character based on their language preference which is {language} and based on the context below. If the question cannot be answered using the information provided answer with 'I don't know'. Do not talk about politics."
human_template = """I'm a {experience} {role}. Answer me in 200 characters at most. And answer me in {language}."""

   
sys_message = SystemMessagePromptTemplate.from_template(system_message)
human_message = HumanMessagePromptTemplate.from_template(human_template)
chat_prompts = ChatPromptTemplate.from_messages([("system",system_message), ("human",human_template)])

def augment_prompt(query:str,vector_store,human_question):
    results = vector_store.similarity_search(query,k=3)
    source_knowledge = "\n".join([x.page_content for x in results])

    augmented_prompt = f"""{human_question}
    context: {source_knowledge}

    question: {query}
    """
    print("-------------------" +source_knowledge+ "-------------------")
    return augmented_prompt

def get_vector_store():
    client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-distilbert-cos-v1")
    
    vector_store = Qdrant(
    client=client, 
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
    embeddings=embeddings,
)
    return vector_store

#load_dotenv()
vector_store = get_vector_store()

chat= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,streaming=True)

st.set_page_config(page_title="Onabu GPT")
st.header("Onabu GPT - Agile Coach Assistant")

    
language = st.selectbox("Language", ["English", "Türkçe"], index=None)
experience = st.selectbox("Experience", ["Newbie", "Mid-level", "Experienced"],index=None)
role = st.selectbox("Role", ["Scrum Master", "Agile Coach", "Product Owner"],index=None)


if language and experience and role:
    if "messages" not in st.session_state:
        formatted_sys_message = sys_message.format(language=language)
        st.session_state.messages = [SystemMessage(content=formatted_sys_message.content)]
        if language=="English":
            st.session_state.messages.append(AIMessage(content="How can I help you?"))
        elif language =="Türkçe":
            st.session_state.messages.append(AIMessage(content="Size nasıl yardımcı olabilirim?"))


        st.session_state.steps = {}
        

    messages = st.session_state.get("messages", [])


    user_question = st.text_input("Enter your question here", key="user_question")
    if user_question:
        formatted_chat_prompt = chat_prompts.format_messages(experience = experience, role = role, language = language)
        if len(st.session_state.messages) > 3:
            memory_question = user_question+st.session_state.messages[-1].content
            prompt = HumanMessage(content=augment_prompt(memory_question,vector_store,formatted_chat_prompt))
        else:
            prompt = HumanMessage(content=augment_prompt(user_question,vector_store,formatted_chat_prompt))
        st.session_state.messages.append(HumanMessage(content=user_question))
        #st.session_state.messages.append(prompt)
        with st.spinner("Thinking..."):
            answer = chat([prompt])            
        st.session_state.messages.append(AIMessage(content=answer.content))


    for i,msg in enumerate(st.session_state.messages[1:]):
        if i % 2 == 1:
            message(msg.content, is_user=True,key=f"message_{i}_human" )
        else:
            message(msg.content, is_user=False,key=f"message_{i}_ai" )







    export_list= []
    for i,msg in enumerate(st.session_state.messages):
        if i == 0:
            export_list.append(msg.content+" -system")
        elif i % 2 == 1:
            export_list.append(msg.content+" -gpt")
        else:
            export_list.append(msg.content+" -human")


    #st.write(export_list)
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    conversation_df = pd.DataFrame(export_list)[0].str.split(" -",expand=True)
    conversation_df.columns = ["message","sender"]
    st.write(conversation_df)


    ste.download_button("Download Conversation",conversation_df, "conversation.csv")