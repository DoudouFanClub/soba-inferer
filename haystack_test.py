# install rust and cargo from - https://www.rust-lang.org/tools/install
# pip install haystack-ai
# pip install trafilatura
# pip install markdown-it-py mdit_plain

from pathlib import Path

import os
import glob

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

from haystack.components.converters.txt import TextFileToDocument
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.converters.html import HTMLToDocument
from haystack.components.converters.markdown import MarkdownToDocument

class SobaRAG:
    def __init__(self, data_directory = '', document_store_directory = ''):
        # Retrieve either '/data/' directory or '/document/doc.json' directory
        if data_directory != '':
            self.document_store = InMemoryDocumentStore()

            # Retrieve 
            self.pipeline = Pipeline()
            files = list( Path(os.path.dirname(__file__) + '\\data\\').rglob("*.*") )

            file_dict = self.filter_file_lists(files)
            for file_ext, file_list in file_dict.items():
                match file_ext:
                    case '.md':
                        self.initialize_reader(file_list, 'md_to_document',   MarkdownToDocument)
                        self.run_reader(file_list, 'md_to_document')
                    case '.txt':
                        self.initialize_reader(file_list, 'text_to_document', TextFileToDocument)
                        self.run_reader(file_list, 'text_to_document')
                    case '.pdf':
                        self.initialize_reader(file_list, 'pdf_to_document',  PyPDFToDocument)
                        self.run_reader(file_list, 'pdf_to_document')
                    case '.html':
                        self.initialize_reader(file_list, 'html_to_document', HTMLToDocument)
                        self.run_reader(file_list, 'html_to_document')

        elif document_store_directory != '':
            self.document_store = InMemoryDocumentStore().load_from_disk(document_store_directory)

        else:
            raise ValueError('Requires at least a "data_directory" parameter or "document_store_directory"')


    def filter_file_lists(self, file_list):
        # Iterate all extensions, for each extension, check if extension exists within file in file_list
        # if it exists, store within extension's file list
        valid_extensions = ['.md', '.txt', '.pdf', '.html']
        file_dict = { ext: [file for file in file_list if ext in str(file)] for ext in valid_extensions }
        return file_dict


    def initialize_reader(self, file_list, name, instance):
        if len(file_list) > 0:
            self.pipeline.add_component(name=name, instance=instance())
            self.pipeline.add_component(name=f"{name}_cleaner",  instance=DocumentCleaner())
            self.pipeline.add_component(name=f"{name}_splitter", instance=DocumentSplitter(split_by="sentence", split_length=10, split_overlap=2))
            self.pipeline.add_component(name=f"{name}_writer",   instance=DocumentWriter(document_store=self.document_store))

            print(f"{name}.documents")
            print(f"{name}_cleaner.documents")

            self.pipeline.connect(f"{name}.documents",          f"{name}_cleaner.documents")
            self.pipeline.connect(f"{name}_cleaner.documents",  f"{name}_splitter.documents")
            self.pipeline.connect(f"{name}_splitter.documents", f"{name}_writer.documents")
        else:
            print(name, ' not initialized, list does not contain any files')


    def run_reader(self, file_list, name):
        if len(file_list) > 0:
            self.pipeline.run({ name: {'sources': file_list} })
        else:
            print(name, ' no files needed to convert to document')




if __name__== "__main__":
    rw_path = os.path.dirname(__file__) + '\\data\\'
    soba = SobaRAG(data_directory=rw_path)
    print(soba.document_store.storage)
    soba.document_store.save_to_disk(rw_path + 'document.json')