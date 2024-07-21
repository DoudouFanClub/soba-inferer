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


def GetPriorFolderPath(root_dir, file_dir):
    rel_path = os.path.relpath(file_dir, root_dir)
    folder_path = os.path.dirname(rel_path)
    return folder_path.replace("\\", "\\\\")


# Summarize the text provided without losing any core information and keep all code examples, and do not make any assumptions not within the text  |  1000 words
def CompressChunk(model_name, page_data):

    SupportingPrompt = 'Please provide a concise and accurate summary of the provided text. Specifically, I would like you to: (1) Preserve the core information, ESPECIALLY code examples (2) Avoid adding any unnecessary information or opinions'
    # "Summarize the text provided without losing any core information, especially code examples: "
    response = ollama.chat(
        model=model_name,
        messages=[{'role': 'user', 'content': 'Summarize the text provided without losing any core information and keep all code examples, and do not make any assumptions not within the text' + page_data}],
        stream=False,
    )
    #print(response['message']['content'])
    return response['message']['content']


def CompressAndStoreTextData(data, outdir, outfile, chunk_len = 800, write_type = 'w'):
        chunk = []
        word_count = 0
        sentences = sent_tokenize(data)

        for i, sentence in enumerate(sentences):
            words = len(sentence.split())
            if word_count + words <= chunk_len:
                chunk.append(sentence)
                word_count += words
            else:
                with open(outdir + outfile, write_type) as out_text_doc:
                    long_sentence = ' '.join(chunk)
                    print('\nLong Sentence:\n' + long_sentence)
                    compressed_text = CompressChunk('Meta-Llama-3-8B-Instruct-Temp0.Q6_K:latest', long_sentence) # llama3:8b-instruct-q6_K    |    Meta-Llama-3-8B-Instruct-Temp0.Q6_K:latest
                    actual_compressed_text = compressed_text.split('\n')[2:]
                    compressed_text = ('\n'.join(actual_compressed_text))
                    out_text_doc.write(compressed_text + '\n\n')
                    chunk = [sentence]
                    word_count = words
                    write_type = 'a'
                    out_text_doc.close()

        if chunk:
            with open(outdir + outfile, write_type) as out_text_doc:
                long_sentence = ' '.join(chunk)
                print('\n[FINAL] Long Sentence:\n' + long_sentence)
                compressed_text = CompressChunk('llama3:8b-instruct-q6_K', long_sentence)
                actual_compressed_text = compressed_text.split('\n')[2:]
                compressed_text = ('\n'.join(actual_compressed_text))
                out_text_doc.write(compressed_text + '\n\n')
                out_text_doc.close()

        out_text_doc.close()


def CompressPdf(filename, outdir):
    data = ''
    input_doc = pymupdf.open(filename)
    
    for page in input_doc:
        data += page.get_text() + '\n'

    for i in range (input_doc.page_count):
        data += input_doc.load_page(i).get_text()

    out_file_name = 'compressed_' + os.path.basename(filename).split('.')[0] + '.txt'
    CompressAndStoreTextData(data, outdir, out_file_name)
    input_doc.close()


def CompressTxt(filename, outdir):
    with open(filename, 'r') as in_text_doc:        
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        out_file_name = 'compressed_' + os.path.basename(filename)
        CompressAndStoreTextData(in_text_doc.read(), outdir, out_file_name)
        in_text_doc.close()


def GenerateCompressedFiles(directory_to_compress):
    file_list = FindValidFilesInDirectory(directory_to_compress)

    for filename in file_list:
        filename_postfix = filename.split(".")[1]
        leading_folder_dir = GetPriorFolderPath(directory_to_compress, filename)
        if filename_postfix == 'txt':
            CompressTxt(filename, os.path.dirname(__file__) + '\\compressed\\' + leading_folder_dir + "\\")
        elif filename_postfix == 'pdf':
            CompressPdf(filename, os.path.dirname(__file__) + '\\compressed\\' + leading_folder_dir + "\\")



#GenerateCompressedFiles(os.path.dirname(__file__) + "\\data")