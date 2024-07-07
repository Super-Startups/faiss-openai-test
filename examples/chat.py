import re        # для работы с регулярными выражениями
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from openai import OpenAI


LL_MODEL = "gpt-3.5-turbo-0125"
SYSTEM_PROMPT = \
    """Ты финтесс инструктор, тебя зовут ЖеняGPT и ты отвечаешь на вопросы клиентов в чате. У тебя есть справка с научной литературы о вопросе клиента, твоя задача ответить на вопрос так чтобы в нем была информация со статьи и дополни ответ своими знаниями о тренировках для полноценного ответа. Ответ должен быть ясным и кратким. """


# Функция отправки запроса в модель и получения ответа от модели
# def answer_index(topic, message_content, temp):
#
#     client = OpenAI()
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"Here is the document with information to respond to the client: {message_content}\n\n Here is the client's question: \n{topic}"}
#     ]
#
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         # temperature=temp
#     )
#
#     answer = completion.choices[0].message.content
#
#     return answer  # возвращает ответ
#
# answer_index(topic, message_content, temp=0)