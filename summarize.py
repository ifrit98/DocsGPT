
import os
import pickle
import argparse

import faiss
from gpt_index.readers import SimpleDirectoryReader
from gpt_index.readers.schema.base import Document as GPTIndexDocument
from langchain.docstore.document import Document as LangChainDocument

from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain

import dotenv
dotenv.load_dotenv()
# os.environ['OPENAI_API_KEY'] = args.openai_api_key
embeddings = OpenAIEmbeddings()



def get_documents(text_dir, verbose=False):
    loader = SimpleDirectoryReader()
    documents = loader.load_data(text_dir)
    if verbose:
        print(len(documents))
        print(documents[0].text[:400])
    return documents
    

def save_faiss_store(store, outpath):
    faiss.write_index(store.index, os.path.join(outpath, "docs.index"))
    store.index = None
    with open(os.path.join(outpath, "faiss_store.pkl"), "wb") as f:
        pickle.dump(store, f)


def load_faiss_store(basepath):
    index = faiss.read_index(os.path.join(basepath, "docs.index"))
    with open(os.path.join(basepath, "faiss_store.pkl"), "rb") as f:
        store = pickle.load(f)
    store.index = index
    return store


def add_texts_to_search_index(texts, search_index):
    search_index.add_texts(texts)
    return search_index

# TODO: add recursive and other text splitter option
def create_index(documents, chunk_size=1000, chunk_overlap=50, sep="\n"):
    source_chunks = []
    splitter = CharacterTextSplitter(
        separator=sep, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for source in documents:
        for chunk in splitter.split_text(source.text):
            source_chunks.append(LangChainDocument(page_content=chunk))

    search_index = FAISS.from_documents(source_chunks, embeddings)
    return search_index


def summarize(texts, temperature=0, max_tokens=1024, chain_type="map_reduce"):
    llm = OpenAI(temperature=temperature, max_tokens=max_tokens)
    summary_chain = load_summarize_chain(llm, chain_type=chain_type)
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
    print("Running summarize document chain...")
    return summarize_document_chain.run(texts)


def blog_post(topic,
              documents=None,
              search_index=None,
              max_tokens=1024,
              prompt_template=None):

    if documents is not None and search_index is None:
        source_chunks = []
        splitter = CharacterTextSplitter(separator=". ", chunk_size=1024, chunk_overlap=0)
        for source in documents:
            for chunk in splitter.split_text(source.text):
                source_chunks.append(LangChainDocument(page_content=chunk))

        search_index = FAISS.from_documents(source_chunks, embeddings)

    elif documents is None and search_index is None:
        raise ValueError("Must pass at least `documents` or `search_index`")

    prompt_template = """Use the context below to write a 500 word blog post about the topic below:
        Context: {context}
        Topic: {topic}
        Blog post:"""

    # TODO: Engineer prompt so we can get detailed blog outputs, like a badass summary with notes...
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "topic"]
    )
    llm = OpenAI(temperature=0, max_tokens=max_tokens)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    def generate_blog_post(topic):
        docs = search_index.similarity_search(topic, k=4)
        inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
        results = chain.apply(inputs)
        print("{} Blog:\n".format(topic), results)
        return results

    return generate_blog_post(topic)


# parser = argparse.ArgumentParser(description='Process some arguments.')
# parser.add_argument("-d", '--text_dir', help='Directory where text files are located') # default='https://www.youtube.com/watch?v=GVRDGQhoEYQ',
# parser.add_argument("-i", '--index_outpath', default='./huberman_index', help='Path to save the vectorstore.')
# parser.add_argument("-l", '--index_inpath', default='./huberman_index', help='Path to load the vectorstore.')
# parser.add_argument("-o", '--openai_api_key', default="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", help='OpenAI API Key')
# parser.add_argument("-t", "--temperature", default=0.1, type=float)
# parser.add_argument("-s", "--sep", default="\n", type=str)
# parser.add_argument("-cs", "--chunk_size", default=1000, type=int)
# parser.add_argument("-co", "--chunk_overlap", default=0, type=int)
# parser.add_argument("-mt", "--max_tokens", default=256, type=int)
# args = parser.parse_args()


# os.environ['OPENAI_API_KEY'] = args.openai_api_key
# embeddings = OpenAIEmbeddings()


# # Load the search index from disk or create it from documents
# if os.path.exists(args.index_inpath):
#     search_index = load_faiss_store(args.index_inpath)
#     documents = None
# else:
#     documents = get_documents(args.text_dir)
#     search_index = create_index(documents)
#     print(search_index)
#     save_faiss_store(search_index, args.index_outpath)


# # Write a 500 word blog given some topic title and the search_index
# if args.blog_mode:
#     blogs = blog_post(topic=input("Blog topic: "), search_index=search_index)
#     with open(args.blog_outpath, 'w') as f:
#         f.write("BLOG RESULTS:\n\n")
#     with open(args.blog_outpath, 'a') as f:
#         for blog in blogs:
#             f.write("\n\n ---------- NEXT BLOG ----------- \n\n")
#             f.write(blog['text'])

# # Write a summary of the documents contained in the search_index
# if args.summary_mode:
#     if documents is None:
#         documents = get_documents()

#     texts = list(map(lambda x: x.text, documents))
#     summaries = list(map(lambda text: summarize(texts=[text], max_tokens=500), texts))
#     print(summaries)

#     with open(args.summary_outpath, 'w') as f:
#         f.write("SUMMARY RESULTS:\n\n")

#     with open(args.summary_outpath, 'a') as f:
#         for summ in summaries:
#             f.write("\n\n ---------- NEXT SUMMARY ----------- \n\n")
#             f.write(summ)



###########################################################################################
###########################################################################################


# # Setup LLM and chunk text
# llm = OpenAI(temperature=0.1, max_tokens=600)
# text_splitter = CharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=50, separator="\n")
# texts = text_splitter.split_text(text)

# # Create langchain docs (text wrapper)
# docs = [Document(page_content=t) for t in texts]

# Here we load in the data in .txt format and split accordingly
from pathlib import Path
def extract_data(data_path, chunk_size=1000, chunk_overlap=50, sep="\n", encoding="utf-8"):
    ps = list(Path(data_path).glob("**/*.rst"))
    ps = ps + list(Path(data_path).glob("**/*.md"))
    ps = ps + list(Path(data_path).glob("**/*.txt"))

    data = []
    sources = []
    for p in ps:
        with open(p, encoding=encoding) as f:
            data.append(f.read())
        sources.append(p)

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=sep
    )
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))
    return docs, metadatas


# docs, metas = extract_data(r"C:\Users\stgeorge\Desktop\micro")

# Create a prompt template for summary report (detailed)
prompt_template = """Write a 500 word summary report about the following text, first generate a 1-2 paragraph summary, then clearly delineate and explain the important concepts and ideas discussed in detail:
{text}
SUMMARY REPORT:"""
# TODO: THIS IS BROKEN AS FUCK.  MAKE SURE CHAR SPLIT WORKS WELL...

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
# summary = summarize(docs)
# print("SUMMARY: {}".format(summary))
# with open("youtube_summary.txt", "w") as f:
#     f.writelines(summary)


def get_topics(docs):
    prompt_template = """Extract at the 5 key topics discussed in the following text and return as a list:
    {text}
    TEXT: "A string of text example"
    TOPICS LIST:
    - Topic 1
    - Topic 2
    - Topic 3
    - Topic 4
    - Topic 5
    TOPICS LIST:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    topics = summarize(docs, PROMPT)
    topics_list = topics.split("- ")

    if topics_list[0] == '\n': topics_list = topics_list[1:]
    topics_list = list(map(lambda x: x.split("\n")[0], topics_list))

    return [t for t in topics_list if t != ''] # Remove null strings

# topics_list = get_topics(docs)
# print("Topics: {}".format(topics_list))
# with open("extracted_topics.txt", "w") as f:
#     for topic in topics_list:
#         f.write(topic + "\n")




def get_blog_post(topic,
                  k=4,
                  documents=None,
                  search_index=None,
                  max_tokens=512,
                  prompt_template=None):              

    if documents is not None and search_index is None:
        source_chunks = []
        splitter = CharacterTextSplitter(separator=". ", chunk_size=1024, chunk_overlap=0)
        for source in documents:
            for chunk in splitter.split_text(source.text):
                source_chunks.append(LangChainDocument(page_content=chunk))

        search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

    elif documents is None and search_index is None:
        raise ValueError("Must pass at least `documents` or `search_index`")

    prompt_template = prompt_template or """Use the context below to write a 500 word blog post about the topic below:
        Context: {context}
        Topic: {topic}
        Blog post:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "topic"]
    )
    llm = OpenAI(temperature=0, max_tokens=max_tokens)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    def generate_blog_post(topic):
        docs = search_index.similarity_search(topic, k=k)
        inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
        results = chain.apply(inputs)
        print("{} Blog:\n".format(topic), results)
        return results

    return generate_blog_post(topic)


def build_index_and_write_blogs(docs, topics_list, index_path='index.pkl', save=True):
    if index_path is not None and os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
    else:
        index = FAISS.from_documents(docs, OpenAIEmbeddings())
    
    if save:
        with open(index_path, 'wb') as f:
            pickle.dump(index, f)

    blogs = []
    for i, topic in enumerate(topics_list):
        print("Writing blog for topic: {}".format(topic))
        post = get_blog_post(topic, search_index=index)
        blogs.append(post[0]) # maybe unlist
        print("Completed!")

    for i, blog in enumerate(blogs):
        with open('blog_{}.txt'.format(i), 'w') as f:
            f.write(blog['text'])

    return blogs

# blogs = build_index_and_write_blogs(docs, topics_list)
# print("Done writing and saving blogs!")