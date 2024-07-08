import json
import sys

import tiktoken

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QLineEdit, QTextEdit, QSplitter, \
    QHBoxLayout, QFileDialog, QMessageBox, QDialog
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

import constants
from index_utils import create_index_files, load_index, get_material


EMBEDDINGS_MODEL = "text-embedding-3-small"

initial_prompt = """Ты фитнес-инструктор, тебя зовут ЖеняGPT и ты отвечаешь на вопросы клиентов в чате.
Твоя задача – ответить на вопросы, основываясь на литературе и добавляя свои знания о тренировках.
Отвечай максимально тезисно и коротко.
Если тебе нужно больше информации, попроси у клиента уточнения.
Если ты не знаешь ответа, скажи об этом клиенту и попроси уточнения."""

index = None

open_ai_client = OpenAI(api_key=constants.API_KEY)

model = "gpt-3.5-turbo"

messages = []


def get_tokens_count(input_text):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(input_text)
    return len(tokens)


def get_system_message():
    return {"role": "system", "content": prompt_field.toPlainText()}


def ask_gpt_with_context(context_messages):
    completion = open_ai_client.chat.completions.create(
        model=model,
        messages=context_messages,
    )

    answer = completion.choices[0].message

    return answer


def on_create_index():
    options = QFileDialog.Options()
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.AnyFile)
    file_name, _ = file_dialog.getOpenFileName(window, "Open File", "",
                                               options=options)
    if file_name:
        create_index_files(file_name, constants.API_KEY)


def on_load_index():
    options = QFileDialog.Options()
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.AnyFile)
    file_name, _ = file_dialog.getOpenFileName(window, "Open File", "",
                                               options=options)
    if file_name:
        global index
        index = load_index(file_name, EMBEDDINGS_MODEL, constants.API_KEY)
        print("Index loaded")


def on_send():
    input_text = input_field.toPlainText()

    question_message = "\n Question: \n" + input_text

    if len(messages) == 0:
        messages.append(get_system_message())

    material = ''

    if index:
        material = get_material(input_text, index, 2)
    else:
        output_field.append("\n ERROR: Index is not loaded! \n")

    material_part = ''
    if material:
        material_part = f"Here is the document with information to respond to the client: {material}\n"

    question_part = f"Here is the client's question: \n{input_text} \n"

    full_question = material_part + question_part

    messages.append({"role": "user", "content": question_part})

    messages_to_send = messages[-5:]
    messages_to_send.append({"role": "user", "content": full_question})

    content = '\n'.join([message['content'] for message in messages_to_send])

    tokens_count = get_tokens_count(content)

    answer = ask_gpt_with_context(messages_to_send)

    messages.append({"role": answer.role, "content": answer.content})

    answer_message = f"\n Answer. Tokents used: {tokens_count}: \n" + answer.content

    output_field.append(question_message)
    output_field.append(answer_message)
    input_field.clear()


def on_show_current_messages():
    content = ''

    for message in messages[-5:]:
        content += f"{message['role']}: {message['content']}\n"

    messages_window = QDialog()
    messages_window.setWindowTitle("Current Messages")
    messages_window.setMinimumSize(900, 700)

    messages_layout = QVBoxLayout()
    messages_window.setLayout(messages_layout)

    tokens_count = get_tokens_count(content)
    tokens_count_label = QLabel(f"Approximate tokens count: {tokens_count}")
    messages_layout.addWidget(tokens_count_label)

    messages_label = QLabel("Current Messages")
    messages_layout.addWidget(messages_label)

    messages_content = QTextEdit()
    messages_content.setReadOnly(True)
    messages_layout.addWidget(messages_content)

    messages_content.setText(content)
    messages_window.exec_()


def on_show_materials_inspector():
    materials_inspector_window = QDialog()
    materials_inspector_window.setWindowTitle("Materials Inspector")
    materials_inspector_window.setMinimumSize(900, 700)

    materials_inspector_layout = QVBoxLayout()
    materials_inspector_window.setLayout(materials_inspector_layout)

    materials_inspector_label = QLabel("Materials Inspector")
    materials_inspector_layout.addWidget(materials_inspector_label)

    materials_inspector_content = QTextEdit()
    materials_inspector_content.setReadOnly(True)
    materials_inspector_layout.addWidget(materials_inspector_content)

    materials_inspector_content.setText("Materials Inspector")

    request_editor = QTextEdit()
    request_editor.setPlaceholderText("Enter your question here")
    materials_inspector_layout.addWidget(request_editor)

    get_material_button = QPushButton("Get Material")
    materials_inspector_layout.addWidget(get_material_button)

    def on_get_material():
        question = request_editor.toPlainText()

        if not index:
            materials_inspector_content.setText("ERROR: Index is not loaded!")
            return

        material = get_material(question, index, 2)
        materials_inspector_content.setText(material)

    get_material_button.clicked.connect(on_get_material)
    materials_inspector_window.exec_()


def on_clear_chat():
    messages.clear()
    output_field.clear()
    input_field.clear()


app = QApplication(sys.argv)
window = QWidget()
window.setMinimumSize(800, 600)
window.setWindowTitle("Fitness Chatbot")

left_layout = QVBoxLayout()
right_layout = QVBoxLayout()

output_label = QLabel("Chat")
left_layout.addWidget(output_label)

output_field = QTextEdit()
output_field.setReadOnly(True)
left_layout.addWidget(output_field)

input_label = QLabel("Enter your message:")
left_layout.addWidget(input_label)
input_field = QTextEdit()
left_layout.addWidget(input_field)

button = QPushButton("Send")
button.clicked.connect(on_send)
left_layout.addWidget(button)

create_index_button = QPushButton("Create Index")
create_index_button.clicked.connect(on_create_index)
right_layout.addWidget(create_index_button)

load_index_button = QPushButton("Load Index")
load_index_button.clicked.connect(on_load_index)
right_layout.addWidget(load_index_button)

show_current_messages_button = QPushButton("Show Current Messages")
show_current_messages_button.clicked.connect(on_show_current_messages)
right_layout.addWidget(show_current_messages_button)

show_materials_inspector_button = QPushButton("Show Materials Inspector")
show_materials_inspector_button.clicked.connect(on_show_materials_inspector)
right_layout.addWidget(show_materials_inspector_button)

prompt_label = QLabel("Prompt")
right_layout.addWidget(prompt_label)

prompt_field = QTextEdit()
prompt_field.setText(initial_prompt)
right_layout.addWidget(prompt_field)

clear_chat_button = QPushButton("Clear Chat")
clear_chat_button.clicked.connect(on_clear_chat)
right_layout.addWidget(clear_chat_button)

splitter = QSplitter()

left_widget = QWidget()
left_widget.setLayout(left_layout)
right_widget = QWidget()
right_widget.setLayout(right_layout)

splitter.addWidget(left_widget)
splitter.addWidget(right_widget)

main_layout = QHBoxLayout()
main_layout.addWidget(splitter)

window.setLayout(main_layout)
window.show()
sys.exit(app.exec_())
