
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from pathlib import Path


document_store = InMemoryDocumentStore()
pipeline = Pipeline()
pipeline.add_component("converter", PyPDFToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=5))
pipeline.add_component("writer", DocumentWriter(document_store=document_store))
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "writer")
def convert_pdf_to_document(path):
    pipeline.run({"converter": {"sources": [path]}})




if __name__ == "__main__":
    lst = list(Path("data/pdfs").glob("*.pdf"))
    print(lst)
    # pth = "data/pdfs/shepherd-gruber-2020-the-lean-startup-framework-closing-the-academic-practitioner-divide.pdf"

    # convert_pdf_to_document(pth)