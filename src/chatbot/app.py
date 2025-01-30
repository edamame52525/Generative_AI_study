import streamlit as st
import os
from dotenv import load_dotenv
from chatbot.ChatBot import chatBot
from chatbot.venctor_store_manage import VectorStoreManager


def main():
    # 環境パスの設定
    load_dotenv()

    # APIキーの登録
    api_key = os.getenv("API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key

    # StreamlitでのUI部分
    st.title("中島家bot")

    user_input = st.text_input("質問を入力してください:")
    if st.button("送信"):
        
        response = chatBot(user_message=user_input).create_message()
        st.write(response)

    if st.sidebar.button("ベクトルストア更新"):
        VectorStoreManager(folder_path="chatbot/data")
