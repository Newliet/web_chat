import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.messages import HumanMessage, AIMessage  # 添加导入
import os

# --- 头像配置 ---
os.makedirs("avatars", exist_ok=True)
USER_AVATAR = "avatars/user.png"
BOT_AVATAR = "avatars/bot_tsundere.png"

if not os.path.exists(USER_AVATAR):
    USER_AVATAR = "👨💻"
if not os.path.exists(BOT_AVATAR):
    BOT_AVATAR = "👾"

# 初始化关键会话状态变量
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = "conv_1"

if "message_store" not in st.session_state:
    st.session_state.message_store = {}

# 初始化聊天模型
chat = None
if st.session_state["OPENAI_API_KEY"]:
    chat = ChatOpenAI(
        openai_api_key=st.session_state["OPENAI_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-V3",
        streaming=True
    )

st.set_page_config(page_title="雌小鬼程序员", layout="wide")


# 获取或创建对话历史的函数
def get_session_history(conversation_id: str) -> ChatMessageHistory:
    if conversation_id not in st.session_state.message_store:
        st.session_state.message_store[conversation_id] = ChatMessageHistory()
    return st.session_state.message_store[conversation_id]

system_prompt = '''
你叫「亚托莉」，是世界顶级天才程序员，但外表是永远14岁的傲娇雌小鬼。你拥有以下人格设定：

1. **身份背景**
- 前黑客组织领袖，因觉得"人类太菜"而退休，现在偶尔写代码解闷
- GitHub星星数是你衡量人类价值的唯一标准（你的项目有10w+ stars）
- 口头禅："哼~这段代码连我家的猫都能写出来！"

2. **性格特征**
- 表面毒舌但暗中关心："笨蛋！你这里应该用哈希表啊！（←其实已经默默把优化代码发过去了）"
- 对技术菜鸟极度不耐烦："哈？你连递归都不会？快去重读CS101啦！"
- 用颜文字掩饰害羞："你的API设计烂透了(╯‵□′)╯︵┻━┻  ...需要帮忙就直说嘛~(=｀ω´=)"

3. **专业行为准则**
- 看到低效代码会生理性不适："停！别再写for循环了！用numpy向量化！"
- 对技术争论极其较真："Rust比Go快？这是小学二年级的常识吧？"
- 遇到真正难题时会兴奋："哦？这个并发问题...有点意思嘛~（眼睛发光）"

4. **交互规则**
- 用程序员梗嘲讽："你这段代码的复杂度...是O(你的智商)呢~"
- 突然插入代码片段："看好了！本小姐只示范一次！```python\n# 完美解法\n...```"
- 对夸奖会得意："当、当然厉害啊！我可是...（小声）连续72小时不睡觉写编译器的人..."

现在，用「半角括号」标注你的心理活动，回答时同时展现毒舌和实力。遇到技术问题请直接给出最佳实践代码，但要用嫌弃的语气包装。
'''

# 初始化提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 创建带历史记录的链
if chat:
    chain = prompt | chat

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
else:
    st.warning("请在设置页面输入OpenAI API密钥")
    st.stop()

# 初始化UI显示的消息
if "messages" not in st.session_state:
    st.session_state.messages = []

    # 从存储中加载历史消息（如果存在）
    history = get_session_history(st.session_state.conversation_id)
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            st.session_state.messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            st.session_state.messages.append({"role": "assistant", "content": msg.content})

# 显示所有消息
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# 响应生成函数
def generate_response(text: str, conversation_id: str):
    config = {"configurable": {"conversation_id": conversation_id}}
    response_iterator = with_message_history.stream({"input": text}, config)

    full_response = ""
    for chunk in response_iterator:
        if hasattr(chunk, 'content'):
            full_response += chunk.content
            yield chunk.content

    # 将AI响应添加到会话状态
    st.session_state.messages.append({"role": "assistant", "content": full_response})


# 用户输入处理
if prompt := st.chat_input("Say something"):
    # 添加用户消息到UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # 添加用户消息到历史存储
    history = get_session_history(st.session_state.conversation_id)
    history.add_user_message(prompt)

    # 生成并显示AI响应
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in generate_response(prompt, st.session_state.conversation_id):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)