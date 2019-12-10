import codecs
import re
from nltk import word_tokenize

def preprocess_wiki(read_path='.../corpora/wikipedia/wiki-en.txt', write_path='.../wiki-en-pre2.txt'):
    """
    >>> preprocess_wiki()
    """
    # with codecs.open(read_path, 'r', 'utf8') as inp:
    #     count = 0
    #     for i, line in enumerate(inp.readlines()):
    #         count = i + 1
    #     inp.close()
    #     print("Total number of lines: %d", count)
    with codecs.open(read_path, 'r', 'utf8') as inp:
        text = inp.read()
        inp.close()
    # this is the regex for removing the wiki page markers
    (text, n) = re.subn(r'^\[\[\d+\]\]\n', '', text, flags=re.MULTILINE)
    print("Number of replacements made: %d", n)

    with codecs.open(write_path, 'w', 'utf8') as outp:
        outp.write(text)
        outp.close()

    # with codecs.open(write_path, 'r', 'utf8') as inp:
    #     count = 0
    #     for i,line in enumerate(inp.readlines()):
    #         count = i + 1
    #     inp.close()
    #     print("Total number of lines after replacements: %d", count)

def main():
    preprocess_wiki()

if __name__ == "__main__":
    main()

