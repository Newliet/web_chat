import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
import os

'''
您的代码已经接近支持上下文对话，但存在几个关键问题导致上下文记忆失效。主要问题在于：

对话历史存储位置错误（使用普通字典而非会话状态）

初始化逻辑顺序问题

消息格式不匹配
'''

# --- 头像配置 ---
os.makedirs("avatars", exist_ok=True)
USER_AVATAR = "avatars/user.png"
BOT_AVATAR = "avatars/bot_tsundere.png"

if not os.path.exists(USER_AVATAR):
    USER_AVATAR = "👨💻"
if not os.path.exists(BOT_AVATAR):
    BOT_AVATAR = "👾"

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
else:
    chat = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"],
                      base_url="https://api.siliconflow.cn/v1",
                      model="deepseek-ai/DeepSeek-V3",
                      streaming=True)

st.set_page_config(page_title="雌小鬼程序员", layout="wide")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，回答用户问题并提供解决办法。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

if chat:
    chain = prompt | chat
else:
    st.warning("Please set your OpenAI API key in the settings page.")
    st.stop()

store = {}

def get_session_history(conversation_id: str) -> ChatMessageHistory:
    key = conversation_id
    if key not in store:
        store[key] = ChatMessageHistory()
    history = store[key]
    if len(history.messages) > 10:
        history.messages = history.messages[-10:]
    return history

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "conv_1"

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="对话唯一标识符",
            default=st.session_state.conversation_id,
            is_shared=True,
        )
    ]
)

def generate_response(text: str, conversation_id: str):
    config = {"configurable": {"conversation_id": conversation_id}}
    response_iterator = with_message_history.stream({"input": text}, config)
    for chunk in response_iterator:
        yield chunk.content

if chat:
    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            response_placeholder = st.empty()
            full_response = ""

            for chunk in generate_response(prompt, st.session_state.conversation_id):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    with st.container():
        st.warning("Please set your OpenAI API key in the settings page.")

