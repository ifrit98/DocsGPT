import pickle
from typing import Optional, Tuple

import gradio as gr
from query_data import get_chain
from threading import Lock

# NEED langchain == 0.0.87
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings

from modules.file import SimpleDirectoryReader

import dotenv
dotenv.load_dotenv()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_directory', type=str, help='Path to directory containing data files to index')
parser.add_argument('-t', '--temperature', type=float, default=0.7, help='LLM temperature setting... lower == more deterministic')
parser.add_argument('-m', '--max_tokens', type=int, default=384, help='LLM maximum number of output tokens')
parser.add_argument('-v', '--vectorstore_path', default="vectorstore.pkl", type=str, help='Path to saved index')
parser.add_argument('-dv', '--docs_vectorstore_path', default="vectorstore_from_docs.pkl", type=str, help='Path to save temporary index')
parser.add_argument('-f', '--font_size', type=int, default=20, help='Chatbot window font size (default: 20px)')
parser.add_argument('-p', '--public', action='store_true', default=False, help="Make a public, sharable link...")
args = parser.parse_args()


DOCS_VECTORSTORE_PATH = args.docs_vectorstore_path
DATA_DIRECTORY = args.data_directory
NEW_VECTORSTORE_SET = False
NEW_DIRECTORY_SET = False


# Attempt to load base vectorstore
try:

    with open(args.vectorstore_path, "rb") as f:
        VECTORSTORE = pickle.load(f)

    print("Loaded vectorstore from `{}`.".format(args.vectorstore_path))

    chain = get_chain(VECTORSTORE, temperature=args.temperature, max_tokens=args.max_tokens)

    print("Loaded LangChain...")

except:

    VECTORSTORE = None

    print("NO vectorstore loaded. Flying blind")



def create_index(dirpath, 
                 chunk_size=1000, 
                 chunk_overlap=25, 
                 save_index=True, 
                 index_path='live_vectorstore.pkl'):

    # Guesstimate cost
    def estimate_cost(docs, char_per_penny=126556):
        c = 0
        for d in docs: c += len(d.page_content)
        cost = (c / char_per_penny) # in cents
        dollars = int(cost / 100)
        cents = str(cost).split(".")[-1][:2]
        print("Estimated cost ($): " + str(dollars) + "." + cents)
        return cost

    # Load data
    loader = SimpleDirectoryReader(dirpath, recursive=True, exclude_hidden=True)
    documents = loader.load_data()

    # Convert to langchain format
    docs = []
    for doc in documents:
        docs.append(doc.to_langchain_format())

    # Get lengths
    doc_lengths = lambda x: list(map(lambda x: len(x.page_content), x))
    print(doc_lengths(docs))
    print("Max document length (before chunking): {}".format(max(doc_lengths(docs))))

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    print("Max split document length (after chunking): {}".format(max(doc_lengths(split_docs))))
    print("# Split docs:", len(split_docs))

    _ = estimate_cost(split_docs)

    # Load Data to vectorstore
    print("Creating vectorstore from split documents and OpenAI embeddings...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    if save_index:
        # Save the vectorstore
        with open(index_path, 'wb') as f:
            pickle.dump(vectorstore, f)

    return vectorstore    


def set_directory(directory: str):

    if directory is not None:

        global DATA_DIRECTORY, NEW_DIRECTORY_SET

        DATA_DIRECTORY = r"{}".format(directory)

        print("Directory from inside set_directory():", DATA_DIRECTORY)

        create_index(directory, index_path=DOCS_VECTORSTORE_PATH)

        set_vectorstore(DOCS_VECTORSTORE_PATH)

        global VECTORSTORE
        chain = get_chain(VECTORSTORE, temperature=args.temperature, max_tokens=args.max_tokens)

        NEW_DIRECTORY_SET = True

        return chain


def initialize_chain():
    chain = get_chain(VECTORSTORE, temperature=args.temperature, max_tokens=args.max_tokens)
    print("LangChain initialized!")
    return chain


def set_vectorstore(vectorstore_path: str):

    global VECTORSTORE, NEW_VECTORSTORE_SET

    if vectorstore_path is not None:

        try:
            with open(vectorstore_path, "rb") as f:
                VECTORSTORE = pickle.load(f)

            print("Loaded `{}`".format(vectorstore_path))
            NEW_VECTORSTORE_SET = True

            chain = get_chain(VECTORSTORE, temperature=args.temperature, max_tokens=args.max_tokens)

        except:
            VECTORSTORE = None
            NEW_VECTORSTORE_SET = False
            print("NO vectorstore loaded. Reverting to original {}".format('vectorstore.pkl'))    

        return chain


class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
        
    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]], chain, #, dirpath: Optional[str], vectorstore_path: Optional[str],
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []

            # If chain is None, that is because it's the first pass and user didn't press Init.
            if chain is None:
                history.append(
                    (inp, "Please Initialize LangChain by clikcing 'Start Chain!'")
                )
                return history, history
        
            # Run chain and append input.
            output = chain({"question": inp, "chat_history": history})["answer"]
            history.append((inp, output))

        except Exception as e:
            raise e

        finally:
            self.lock.release()

        return history, history


chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>ChatGPT with your data</center></h3>")

        directory_textbox = gr.Textbox(
            placeholder="Enter a directory path",
            show_label=False,
            lines=1
        )
        submit_directory = gr.Button(value="Submit", variant="secondary").style(full_width=False)

    with gr.Row():
        vectorstore_textbox = gr.Textbox(
            placeholder="Enter a vectorstore path",
            show_label=False,
            lines=1
        )
        submit_vectorstore = gr.Button(value="Load", variant="secondary").style(full_width=False)

    chatbot = gr.Chatbot().style(font_size=args.font_size)

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about currently the loaded vectorstore",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    with gr.Row():
        init_chain_button = gr.Button(value="Start Chain!", variant="primary").style(full_width=False)

    gr.HTML("Please initialize the chain by clicking 'Start Chain!' before submitting a question.")
    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗 and Unicorn Farts 🦄💨</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(
        chat, 
        inputs=[message, state, agent_state],
        outputs=[chatbot, state]
    )

    message.submit(
        chat, 
        inputs=[message, state, agent_state],
        outputs=[chatbot, state]
    )

    submit_directory.click(
        set_directory,
        inputs=[directory_textbox],
        outputs=[agent_state],
    )

    submit_vectorstore.click(
        set_vectorstore,
        inputs=[vectorstore_textbox],
        outputs=[agent_state],
    )

    init_chain_button.click(
        initialize_chain,
        inputs=[],
        outputs=[agent_state],
        show_progress=True
    )

block.launch(debug=True, share=args.public)


