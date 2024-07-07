import openai
import re
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
import codecs
import os
import pickle
import faiss


openai.api_key = ""
EMBEDDINGS_MODELS = [
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "text-embedding-3-large",
]


# Функция создания индексной базы знаний
def create_index_db(database, emb_model):
    source_chunks = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048, chunk_overlap=1)

    for chunk in splitter.split_text(database):
      source_chunks.append(Document(page_content=chunk, metadata={}))

    # Инициализирум модель эмбеддингов
    embeddings = OpenAIEmbeddings(model=emb_model)

    db = FAISS.from_documents(source_chunks, embeddings)
    return db


# Функция загруки содержимого текстового файла
def load_text(file_path):
    # Открытие файла для чтения
    with codecs.open(file_path, "r", encoding="utf-8", errors="ignore") as input_file:
        # Чтение содержимого файла
        content = input_file.read()
    return content

# Function to save FAISS index and metadata
def save_faiss_index(index_db, index_path, metadata_path):
    faiss.write_index(index_db.index, index_path)
    with open(metadata_path, 'wb') as f: pickle.dump((index_db.docstore, index_db.index_to_docstore_id), f)


if __name__ == '__main__':
    # Загружаем текст Базы Знаний из файла
    texts = []
    for text_name in os.listdir("data"):
        texts.append(load_text(f"data/{text_name}"))

    os.makedirs('indexes', exist_ok=True)

    # database = load_text('/home/lolik/LLM-RAG/data/4. Развивая силу (полная версия) (1).txt')
    database = "\n\n".join(texts)
    # Создаем индексную Базу Знаний
    for emb_model in EMBEDDINGS_MODELS:
        index_db = create_index_db(database, emb_model)

        save_path = f"indexes/{emb_model}"
        os.makedirs(save_path, exist_ok=True)

        index_path = f"{emb_model}_index.faiss"
        metadata_path = f"{emb_model}_meta.pkl"
        save_faiss_index(index_db, f"{save_path}/index.faiss", f"{save_path}/meta.pkl")