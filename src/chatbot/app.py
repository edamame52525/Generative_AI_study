import streamlit as st
import os
from dotenv import load_dotenv
from chatbot.ChatBot import chatBot
from chatbot.venctor_store_manage import VectorStoreManager


class GUI:

    def __init__(self):

        #.envファイルの読み込み
        load_dotenv()
        # APIキーの登録
        self.api_key = os.getenv("API_KEY")
        self.folder_path = "chatbot/data"
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # 初回処理（UI表示やベクトル化）
        self.vsm = VectorStoreManager(folder_path=self.folder_path)
        self.chatbot = chatBot(self.vsm)
        

    def main(self):
        self.display()


    def display(self):
        # カスタムCSSを追加してサイドバーの色を変更
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                background-color: #F7F390;  /* 落ち着いた黄色 */
            }
            .small-label {
                font-size: 12px;  /* 小さな文字サイズ */
                font-weight: bold;  /* 太字 */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # StreamlitでのUI部分
        st.title("中島家bot")

        # セッション状態の初期化   
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # 入力フォーム
        with st.form(key="qa_form",clear_on_submit=True):
            query = st.text_area("質問を入力してください:", height=100, key="query")
            submit_button = st.form_submit_button("質問する")

        if submit_button and query:
            try:
                # プログレスバーの表示
                with st.spinner("回答を生成中..."):
                    response = self.chatbot.create_message(query)

                # 新しい会話を履歴に追加
                st.session_state.chat_history.append(
                    {"query": query, "answer": response}
                )
                st.write(response)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

        

        st.sidebar.markdown('<p class="small-label">データを学習させる場合はこちらに入力してください</p>', unsafe_allow_html=True)
        learn_text = st.sidebar.text_area("", height=90, key="learn_text")
        if st.sidebar.button("学習させる") and learn_text:
            self.vsm.create_text_data(learn_text)
            self.chatbot.update_knowledge()
            st.sidebar.write("学習が完了しました。")
            # テキストエリアの内容をクリア



