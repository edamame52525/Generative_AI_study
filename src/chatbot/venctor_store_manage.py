import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader


class VectorStoreManager:
    """
    ベクトルストアを管理するクラス
    ・ボタンを押すことによって、loaderを再定義して、ベクトルストアに反映させる機能を作りたい
      ボタンを押す→関数update_vector_store()起動みたいな感じがいいかもしれない。
    
    """
    
    def __init__(self,folder_path:str):
        self.folder_path = folder_path
        self.embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.text_splitter = CharacterTextSplitter(
                                separator="\n\n",
                                chunk_size=150,
                                chunk_overlap=0,
                                )
        self.loader=self.load_text_folder()
        self.index=self.create_vector_store()

    def load_text_folder(self):
        loader = DirectoryLoader(self.folder_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
        return loader
    
 
    def create_vector_store(self):
        print("🔄 ベクトルストアの作成...")

        index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=self.embedding,
        text_splitter=self.text_splitter
        ).from_loaders([self.loader])
        print("✅ ベクトルストア作成完了")

        return index

    def update_vector_store(self):
        self.loader = self.load_text_folder(self.folder_path)
        self.index = self.create_vector_store() 
        st.write("ベクトルストアが更新されました。")
        
    