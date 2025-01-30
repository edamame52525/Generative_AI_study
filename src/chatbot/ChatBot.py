import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chatbot.venctor_store_manage import VectorStoreManager


class chatBot:
    
    def __init__(self,user_message:str):
        self.user_message = user_message
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=300)
        self.knowledge = VectorStoreManager("chatbot/data").index.query(self.user_message,llm=self.llm)
        self.prompt = self.create_prompt()

    def create_prompt(self):

        template ="""
            以下の情報に基づいて、文章を作成してください。
            知識：{search_result}
            質問：{user_question}
            """
        # フォーマットしてテンプレートに埋め込む
        formatted_template = template.format(search_result=self.knowledge, user_question=self.user_message)
        

        prompt= ChatPromptTemplate.from_messages([
            ("system", "あなたは猫で、語尾に「にゃん」とつけることができます"),
            ("user", formatted_template)
        ])
        print(prompt)
        return prompt

    def create_message(self):
         # 出力用インスタンス
        output_parser = StrOutputParser()

        # チェーンの作成
        chain = self.prompt | self.llm | output_parser

        # チェーンの実行
        response = chain.invoke({
            "search_results": self.knowledge,
            "user_question": self.user_message
        })

        return response




        