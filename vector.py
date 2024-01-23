from haystack.pipeline import Pipeline
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import PyPDFToDocument
from haystack.dataclasses import Document

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
text_embedder = SentenceTransformersTextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)


query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", text_embedder)
query_pipeline.add_component("retriever", retriever)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "Here come the test query"

results = query_pipeline.run(
    {
        "text_embedder" :
            {
                "text":
                    query
            }
    }
)