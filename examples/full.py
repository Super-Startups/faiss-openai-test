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



os.environ["OPENAI_API_KEY"] = openai_api_key

index_path = "indexes"
embedding_model = "text-embedding-3-large"

# Функция получения релевантные чанков из индексной базы знаний на основе заданной темы
def get_message_content(topic, index_db, k_num):
    # Поиск релевантных отрезков из базы знаний
    docs = index_db.similarity_search(topic, k = k_num)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### Document excerpt №{i+1}####\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    # print(f"message_content={message_content}")
    return message_content


# Function to load FAISS index and metadata
def load_faiss_index(index_path, metadata_path, embedding_model):
    embedding_function = OpenAIEmbeddings(model=embedding_model)
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        docstore, index_to_docstore_id = pickle.load(f)
    return FAISS(embedding_function, index, docstore, index_to_docstore_id)


if name == '__main__':
    topic = "Какие основные принципы функционального тренинга"
    index_db = load_faiss_index(f"{index_path}/{embedding_model}/index.faiss", f"{index_path}/{embedding_model}/meta.pkl", embedding_model)
    # Ищем реливантные вопросу чанки и формируем контент для модели, который будет подаваться в user
    message_content = get_message_content(topic, index_db, k_num=3)