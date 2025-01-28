import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time
import requests

# 環境パスの設定
load_dotenv()

# APIキーの登録
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# LLMのインスタンス作成
llm = ChatOpenAI(model_name = "gpt-4o-mini",temperature = 0,max_tokens=100)


def send_message2llm(user_input):
        template = """
        あなたは校正者です。今から書いた文章が間違っていれば、正しく直してください。
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

        






