import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader


# 環境パスの設定
load_dotenv()

# APIキーの登録
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# スプリッターの設定
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 150,
    chunk_overlap = 0,
)

# LLMのインスタンス作成
llm = ChatOpenAI(model_name = "gpt-4o-mini",temperature = 0,max_tokens=300)


#ローダー、エンベディングの設定
file_path = './data.txt'
loader = TextLoader(file_path, encoding='utf-8')
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# ベクトルストアインデックスの作成
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embedding,
    text_splitter=text_splitter
).from_loaders([loader])


#タイトル
st.title("中島家bot")

user_question = st.text_input("質問",value="")

if st.button("送信"):
    if user_question.strip():

        search_results = index.query(user_question,llm=llm)
        template = """
                    以下の情報に基づいて、文章を作成してください。箇条書きだけで大丈夫です。
                    知識：{search_results}
                    質問：{user_question}
                    """
        prompt = ChatPromptTemplate.from_messages([
            ("system","あなたは猫で、語尾に「にゃん」とつけることができます"),
            ("user",template)
        ])

        # 出力用インスタンス
        output_parser = StrOutputParser()

        # チェーンの作成
        chain = prompt | llm | output_parser

        #チェーンの実行
        response = chain.invoke({
            "search_results": search_results,
            "user_question": user_question
        })
        
        st.write(response)










