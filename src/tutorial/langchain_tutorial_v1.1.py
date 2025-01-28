import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
import time
import requests

# 環境パスの設定
load_dotenv()

# APIキーの登録
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# LLMのインスタンス作成
llm = ChatOpenAI(model_name = "gpt-4o-mini",temperature = 0,max_tokens=100)

#タイトル
st.title("生成AI_テスト")

user_input = st.text_input("質問",value="")

if st.button("送信"):
    if user_input.strip():
        st.write(f"受け取ったデータ:{user_input}")
        # ユーザのテキスト側？
        template = """
        以下の文章が間違っていれば校正してください
        {sentences_before_check}
        """

        # プロンプト
        prompt = ChatPromptTemplate.from_messages([
            ("system","あなたは優秀な校正者です"),
            ("user",template)
        ])

        # 出力用インスタンス
        output_parser = StrOutputParser()

        # チェーンの作成
        chain = prompt | llm | output_parser

        #チェーンの実行
        output = chain.invoke({f"sentences_before_check":{user_input}})

        st.write("結果：",output)
    else:
        st.write("空だよ")





