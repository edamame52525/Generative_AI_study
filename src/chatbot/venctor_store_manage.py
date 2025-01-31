import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
import os


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
    
    @st.cache_resource
    def create_vector_store(_self):
        print("🔄 ベクトルストアの作成...")
        index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=_self.embedding,
            text_splitter=_self.text_splitter
        ).from_loaders([_self.loader])
        print("✅ ベクトルストア作成完了")
        return index

    def update_vector_store(self):
        st.cache_resource.clear()
        self.loader = self.load_text_folder()
        self.index = self.create_vector_store()
        st.write("ベクトルストアが更新されました。")


    def create_text_data(self, lt: str):
    # まずはテキストの加工
        lt = lt.replace("\n", "")
        chunc = [lt[i:i+150] for i in range(0, len(lt), 150)]
        cn = len(chunc)
        chunc_index = 0

        for root, dirs, files in os.walk(self.folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                with open(file_path, "r+", encoding="utf-8") as f:
                    lines = f.readlines()
                    f.seek(0,2)
                    if lines:
                        while chunc_index < cn:
                            f.write("\n\n" + chunc[chunc_index])
                            print("書き込み１：" + chunc[chunc_index])
                            chunc_index += 1
                    
            
