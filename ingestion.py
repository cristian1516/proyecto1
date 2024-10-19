import os

from dotenv import load_dotenv
load_dotenv()

from consts import INDEX_NAME
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from firecrawl import FirecrawlApp
from langchain.schema import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
def ingest_docs():
    loader = ReadTheDocsLoader("./langchain-docs/api.python.langchain.com/en/latest/", encoding='UTF-8')
    raw_documents = loader.load()
    print(f"loaded{len(raw_documents)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"loaded {len(documents)} documents")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )




def ingest_docs2()-> None:
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])
    url = ("https://es.wikipedia.org/wiki/Porsche, https://newsroom.porsche.com/es/2019/vehiculos/es-porsche-fastest-models-speed-youtube-top-5-series-16897.html#:~:text=%231%20Porsche%20918%20Spyder&text=El%20918%20Spyder%20muestra%20lo,del%20Top%205%20de%20velocidad., https://newsroom.porsche.com/es/2021/historia/PLA-es-ferdinand-porsche-engineering-christophorus-398-25822.html")

    page_content = app.scrape_url(url=url,
                                  params={
                                      "onlyMainContent": True
                                  })
    print(page_content)
    doc = Document(page_content=str(page_content), metadata={"source": url})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents([doc])

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="porsche-proyecto-index"
    )


if __name__ == "__main__":
    ingest_docs2()