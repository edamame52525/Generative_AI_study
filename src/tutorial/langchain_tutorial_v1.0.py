import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import time

load_dotenv()
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(model_name = "gpt-4o-mini",temperature = 0,max_tokens=100)
template = """
次の文章に誤字がないか調べて、誤字があれば訂正してください。
{sentences_before_check}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system","あなたは優秀な校正者です"),
    ("user",template)
])

# 出力用インスタンス
output_parser = StrOutputParser()

# チェーンの作成
chain = prompt | llm | output_parser

#チェーンの実行
print(chain.invoke({"sentences_before_check":"ここんにちは中じまでえす"}))

time.sleep(1)  # 1秒間隔で実行