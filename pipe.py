import os
from pathlib import Path

from haystack.document_stores import WeaviateDocumentStore
from haystack import Pipeline
from haystack.nodes import EmbeddingRetriever, PreProcessor, PDFToTextConverter, PromptNode, AnswerParser, \
    PromptTemplate


def main():
    document_store = WeaviateDocumentStore(
        host="http://localhost",
        port=8080,
        embedding_dim=768
    )

    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True
    )

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/multi--qa-mpnet-base-dot-v1",
        use_gpu=True
    )

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=converter, name="PDFConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["PDFConverter"])
    indexing_pipeline.add_node(component=retriever, name="Retriever", inputs=["PreProcessor"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["Retriever"])

    cwd = Path.cwd()
    pdf_dir = cwd.joinpath("data", "pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))

    indexing_pipeline.run(
        params={"File": pdfs}
    )

    prompt_template = PromptTemplate(
        prompt="Given the provided Documents, answer the Query. Make your answer detailed and long\n Query: {query} Documents: {join(documents)} \n Answer:",
        output_parser=AnswerParser()
    )

    prompt_node = PromptNode(
        model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        default_prompt_template=prompt_template,
        use_gpu=False
    )

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

    query_pipeline.run(
        query="What is a lean canvas",
        params=
        {
            "Retriever":
                {"top_k": 5}
        }
    )


main()
