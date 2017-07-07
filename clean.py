import os
import sys
import logging
import pandas as pd





def normalize_to_unicode(text, encoding="utf-8"):
    if sys.version_info.major == 2:
        if isinstance(text, str):
            text = unicode(text.decode(encoding))
        return text
    else:
        if isinstance(text, bytes):
            text = str(text.decode(encoding))
        return text

def normalize_to_bytes(text, encoding="utf-8"):
    if sys.version_info.major == 2:
        if isinstance(text, unicode):
            text = str(text.encode(encoding))
        return text
    else:
        if isinstance(text, str):
            text = bytes(text.encode(encoding))
        return text

def convert_windows1252_to_utf8(text):
    return text.decode("cp1252")

def add_newline_to_unicode(text):
    return text + u"\n"

def single_item_process_standard(text):
    text = normalize_to_unicode(text).strip()
    text = add_newline_to_unicode(text)
    return normalize_to_bytes(text)

def single_item_process_funky(text):
    text = convert_windows1252_to_utf8(text).strip()
    text = add_newline_to_unicode(text)
    return normalize_to_bytes(text)


def process(lines):
    out_lines = []
    for line in lines:
        try:
            out_lines.append(single_item_process_standard(line))
        except UnicodeDecodeError:
            out_lines.append(single_item_process_funky(line))
    return out_lines
            
def load_dataset(filename):
    with open(filename, "rb") as fp:
        lines = fp.readlines()
    lines = process(lines)
    path_part, ext_part = os.path.splitext(filename)
    new_filename = "{}_cleaned{}".format(path_part, ext_part)
    with open(new_filename, "wb") as fp:
        fp.writelines(lines)
    #logging.info('Cleaning Files done')
    
def load_files(filename):
    out = []
    with open(filename, "rb") as fp:
        for line in fp.readlines():
            try:
                out.append(normalize_to_unicode(line).strip())
            except UnicodeDecodeError:
                logger.exception('Broken line: {}'.format(line))
    return out

def pandas_dataframe(filename):
    return pd.DataFrame.from_csv(filename)

if __name__ == '__main__':
    load_dataset('manual_dataset.csv')
    load_dataset('google_asr_dataset.csv')
    load_dataset('ibm_asr_dataset.csv')
    load_dataset('ms_asr_dataset.csv')
    logging.info('Cleaning Done........')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] '
        '{%(funcName)s:%(lineno)d} %(message)s')

    file_handler = logging.FileHandler('file.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    #logging.info('Cleaning of files done:')
   