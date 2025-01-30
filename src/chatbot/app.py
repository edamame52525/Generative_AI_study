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

    # セッション状態の初期化   
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 入力フォーム
    with st.form(key="qa_form"):
        query = st.text_area("質問を入力してください:", height=100)
        submit_button = st.form_submit_button("質問する")


    if submit_button and query:
        try:
            # プログレスバーの表示
            with st.spinner("回答を生成中..."):
                response = chatBot(user_message=query).create_message()

            # 新しい会話を履歴に追加
            st.session_state.chat_history.append(
                {"query": query,"answer":response}
            )
            st.write(response)

        except:
            st.error()
            
        
        

    if st.sidebar.button("ベクトルストア更新"):
        VectorStoreManager(folder_path="chatbot/data")
