import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chatbot.venctor_store_manage import VectorStoreManager


class chatBot:
    
    def __init__(self,VSM:VectorStoreManager):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=300)
        self.prompt = self.create_prompt()
        self.vsm = VSM
        self.knlg = ""

    def create_prompt(self):

        template ="""
            以下の情報を考慮して、文章を作成してください。
            知識：{search_result}
            質問：{user_question}
            """        

        prompt= ChatPromptTemplate.from_messages([
            ("system", "あなたは中島家を管理するエージェントです。ユーザの質問に対して、シンプルな表現を用いて回答してください。あと、言葉遣いはちょっと優しく"),
            ("user", template)
        ])
        # print(prompt)
        return prompt

    def create_message(self,user_message:str):
        self.knlg = self.vsm.index.query(user_message, llm=self.llm)
        # 出力用インスタンス

        output_parser = StrOutputParser()

        # チェーンの作成
        chain = self.prompt | self.llm | output_parser

        # チェーンの実行
        response = chain.invoke({
            "search_result": self.knlg,  # 変数名を修正
            "user_question": user_message
        })

        return response
    
    
    
    def update_knowledge(self):
        self.vsm.update_vector_store()




