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

# with open("long_text.txt", "w") as f:
#     f.close() #ファイルに書き込むときはこれ使う



"""
long_text = 複数行リテラル
print(len(long_text))
text_list = text_splitter.split_text(long_text)
# print(text_list)
# print(len(text_list))


# ドキュメント化
document_list = text_splitter.create_documents([long_text])
# print(document_list) #確認用
# print(len(document_list)) #確認用


↑のやり方では、テキストデータのリストを作成する。
"""
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 150,
    chunk_overlap = 0,
)

# LLMのインスタンス作成
llm = ChatOpenAI(model_name = "gpt-4o-mini",temperature = 0,max_tokens=300)


"""
# ドキュメントの確認

if not os.path.exists(file_path):
    raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")


# ドキュメントを読み込む
documents = loader.load()
print(documents)
"""

file_path = './data.txt'

loader = TextLoader(file_path, encoding='utf-8')






# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# ベクトルストアインデックスの作成
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=embedding,
    text_splitter=text_splitter
).from_loaders([loader])

"""
.from_documents([document_list])では、データのリストそのものをindexに渡しているが、
.from_loaders(loader)にすれば、いちいちドキュメントを分割せずに利用できるらしい。
"""






user_question = "まりんちゃんについて教えて"
search_results = index.query(user_question,llm=llm)
# print(search_results)
# response =llm.generate(search_results)
# print(response)


template = """
以下の情報に基づいて、文章を作成してください。分かりやすいように伝えてください。
{search_results}

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

print(response)



