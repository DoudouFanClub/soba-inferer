import os
import glob
import nltk
import ollama
import pymupdf
from nltk.tokenize import sent_tokenize

def FindValidFilesInDirectory(directory):
    pdf_file_list = glob.glob(directory + '/**/*.pdf', recursive=True)
    text_file_list = glob.glob(directory + '/**/*.txt', recursive=True)
    return pdf_file_list + text_file_list


def CompressChunk(model_name, page_data):
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': "I want you to summarize the data provided without losing any of its core information, do not add on any assumptions or extra information, and retain all code examples in full: " + page_data}],
        stream=False,
    )
    #print(response['message']['content'])
    return response['message']['content']


def CompressAndStoreTextData(data, outdir, outfile, chunk_len = 800, write_type = 'w'):
    with open(outdir + outfile, write_type) as out_text_doc:
        if not out_text_doc.writable:
            print(outfile + '.txt was unwritable, unable to store compressed data')
            return False
        
        chunk = []
        word_count = 0
        sentences = sent_tokenize(data)

        for i, sentence in enumerate(sentences):
            words = len(sentence.split())
            if word_count + words <= chunk_len:
                chunk.append(sentence)
                word_count += words
            else:
                long_sentence = ' '.join(chunk)
                print('\nLong Sentence:\n' + long_sentence)
                compressed_text = CompressChunk('llama3:8b-instruct-q6_K', long_sentence)
                actual_compressed_text = compressed_text.split('\n')[2:]
                compressed_text = ('\n'.join(actual_compressed_text))
                out_text_doc.write(compressed_text + '\n')
                chunk = [sentence]
                word_count = words

        if chunk:
            long_sentence = ' '.join(chunk)
            print('\n[FINAL] Long Sentence:\n' + long_sentence)
            compressed_text = CompressChunk('llama3:8b-instruct-q6_K', long_sentence)
            actual_compressed_text = compressed_text.split('\n')[2:]
            compressed_text = ('\n'.join(actual_compressed_text))
            out_text_doc.write(compressed_text + '\n')

        out_text_doc.close()


def CompressPdf(filename, outdir):
    data = ''
    input_doc = pymupdf.open(filename)

    if input_doc.is_closed:
        print(filename + ' could not be opened')
        return
    
    for page in input_doc:
        data += page.get_text() + '\n'

    for i in range (input_doc.page_count):
        data += input_doc.load_page(i).get_text()

    out_file_name = 'compressed_' + filename.split('.')[0] + '.txt'
    CompressAndStoreTextData(data, outdir, out_file_name)
    input_doc.close()


def CompressTxt(filename, outdir):
    with open(filename, 'r') as in_text_doc:
        if not in_text_doc.readable:
            print(filename + ' was unreadable, unable to compress')
            return
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        out_file_name = 'compressed_' + filename.split('.')[0] + '.txt'
        CompressAndStoreTextData(in_text_doc.read(), outdir, out_file_name)
        in_text_doc.close()


def GenerateCompressedFiles(directory_to_compress):
    file_list = FindValidFilesInDirectory(directory_to_compress)

    for filename in file_list:
        filename_postfix = filename.split(".")[1]
        if filename_postfix == 'txt':
            CompressTxt(filename, os.path.dirname(__file__) + '\\compressed\\')
        elif filename_postfix == 'pdf':
            CompressPdf(filename, os.path.dirname(__file__) + '\\compressed\\')





# Testing File Directory
#CompressTxt('vrlink_api.txt', os.path.dirname(__file__) + '\\compressed\\')
#CompressPdf('input.pdf', os.path.dirname(__file__) + '\\compressed\\')
#print(FindValidFilesInDirectory( os.path.dirname(__file__)) )