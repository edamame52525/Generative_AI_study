import streamlit as st
from utils import send_message2llm
#タイトル
st.title("生成AI_テスト")

user_input = st.text_input("質問",value="")

if st.button("送信"):
    if user_input.strip():
        send_message2llm(user_input)
        
        # st.write(f"受け取ったデータ:{user_input}")
        # # ユーザのテキスト側？
        # st.write("結果：",output)
