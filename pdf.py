from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pathlib import Path
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.nodes import PreProcessor, EmbeddingRetriever
from haystack.pipeline import Pipeline

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")


converter = PyPDFToDocument(remove_numeric_tables=True, valid_languages=["en"])

pipeline = Pipeline()
pipeline.add_component("converter", converter)
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
pipeline.add_component("writer", DocumentWriter(document_store=document_store))
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "writer")



# preprocessor = PreProcessor()
# retriever = EmbeddingRetriever(
#     document_store = document_store,
#     embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
# )





pipeline.run(
    file_paths=Path("data/pdfs").glob("*.pdf")
)



# print("DONE")
