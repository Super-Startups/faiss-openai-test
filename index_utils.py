from zipfile import ZipFile

import PyPDF2
import faiss
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
import pickle
import re

EMBEDDINGS_MODEL = "text-embedding-3-small"


def load_faiss_index(index_path, metadata_path, embedding_model, api_key):
    embedding_function = OpenAIEmbeddings(model=embedding_model, api_key=api_key)
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        docstore, index_to_docstore_id = pickle.load(f)
    return FAISS(embedding_function, index, docstore, index_to_docstore_id)


def get_material(topic, index, k_num):
    docs = index.similarity_search(topic, k=k_num)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join(
        [f'\n#### Document excerpt №{i + 1}####\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    return message_content


def load_index(index_path, embedding_model, api_key):
    base_path = '/'.join(index_path.split('/')[:-1])
    base_name = index_path.split('/')[-1]
    name = base_name.replace('_index.faiss', '')

    metadata_file_name = f'{name}_metadata.pkl'
    metadata_path = f'{base_path}/{metadata_file_name}'

    return load_faiss_index(index_path, metadata_path, embedding_model, api_key)


def create_index_files(path, api_key):
    with ZipFile(path) as zip_file:
        texts = accumulate_texts(zip_file)

    base_path = '/'.join(path.split('/')[:-1])
    base_name = path.split('/')[-1]
    name = base_name.split('.')[0]
    index_path = f'{base_path}/{name}_{EMBEDDINGS_MODEL}_index.faiss'
    metadata_path = f'{base_path}/{name}_{EMBEDDINGS_MODEL}_metadata.pkl'

    index = create_index(texts, EMBEDDINGS_MODEL, api_key)

    save_index(index, index_path, metadata_path)
    print('Index saved at path:' + index_path)


def save_index(index, index_path, metadata_path):
    faiss.write_index(index.index, index_path)
    with open(metadata_path, 'wb+') as f:
        pickle.dump((index.docstore, index.index_to_docstore_id), f)


def create_index(data, model, api_key):
    chunks = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=2048, chunk_overlap=1)

    for chunk in splitter.split_text(data):
        chunks.append(Document(page_content=chunk, metadata={}))

    embedding = OpenAIEmbeddings(model=model, api_key=api_key)

    return FAISS.from_documents(chunks, embedding)


def accumulate_texts(zip_file):
    texts = "\n\n"
    for name in zip_file.namelist():
        with zip_file.open(name) as file:
            if file.name.endswith('.pdf'):
                text = convert_pdf_to_text(file)
                texts = texts + "\n" + text
            else:
                texts = texts + "\n" + file.read().decode('utf-8')
    return texts


def convert_pdf_to_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text
