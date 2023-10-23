import streamlit as st
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
import streamlit_ext as ste



#msgs = StreamlitChatMessageHistory(key="langchain_messages")
#memory = ConversationBufferMemory(chat_memory=msgs ,return_messages=True,output_key="output")

st.title("Onabu GPT")
api_key = st.text_input("Enter OpenAI api key")


text = "You are an experienced agile coach that gives insightful information to customer within 200 character based on their language preference which is {language}."
human_template = """I'm a {experience} {role}. Answer me in 200 characters at most. And answer me in {language}. {message}"""

if api_key:
    model = st.selectbox("OpenAI Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k","text-davinci-003"])
    chat = ChatOpenAI(openai_api_key=api_key, model=model,temperature=0,streaming=True)
    #st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
    sys_message = SystemMessagePromptTemplate.from_template(text)
    human_message = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompts = ChatPromptTemplate.from_messages([("system",text), ("human",human_template)])
    #chain = LLMChain(llm=llm,prompt=chat_prompts,callbacks=[st_cb],memory=memory)

else:
    st.write("Please provide an api key and press enter.")

    


language = st.selectbox("Language", ["English", "Türkçe"], index=None)
experience = st.selectbox("Experience", ["Newbie", "Mid-level", "Experienced"],index=None)
role = st.selectbox("Role", ["Scrum Master", "Agile Coach", "Product Owner"],index=None)

if api_key and language and experience and role:

    if "messages" not in st.session_state:
        formatted_sys_message = sys_message.format(language=language)
        st.session_state.messages = [SystemMessage(content=formatted_sys_message.content)]
        if language=="English":
            st.session_state.messages.append(AIMessage(content="How can I help you?"))
        elif language =="Türkçe":
            st.session_state.messages.append(AIMessage(content="Size nasıl yardımcı olabilirim?"))


        st.session_state.steps = {}
        

    messages = st.session_state.get("messages", [])

    for i,msg in enumerate(st.session_state.messages[1:]):
        if i % 2 == 1:
            message(msg.content, is_user=True,key=f"message_{i}_human" )
        else:
            message(msg.content, is_user=False,key=f"message_{i}_ai" )




    user_input = st.text_input("Enter your message here", key="user_input")
    if user_input:
        formatted_chat_prompt = chat_prompts.format_messages(experience = experience, role = role, language = language,message = user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
            response = chat(formatted_chat_prompt)
        st.session_state.messages.append(AIMessage(content=response.content))

    #prompt = chat_prompts.format_messages(experience = experience, role = role, language = language,message = message)
    #response = chain.run({'role':role, 'message':message, 'experience':experience, 'language':language},callbacks=[st_cb])


   
    
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
    #st.write(st.session_state.messages)
    ste.download_button("Download Conversation",conversation_df, "conversation.cs")