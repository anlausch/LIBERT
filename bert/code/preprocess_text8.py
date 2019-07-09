import codecs
#from spacy.lang.en import English
#import spacy
#import en_core_web_sm
#from nltk import sent_tokenize
from gensim.models import word2vec

def preprocess_text8(read_path='./../data/text8', write_path='./../data/text8_sentences2'):
    """
    >>> preprocess_text8()
    """
    #nlp = spacy.load('en_core_web_sm')
    #nlp = en_core_web_sm.load()
    with codecs.open(read_path, 'r', 'utf8') as inp:
        text = inp.read()
        inp.close()
        #nlp.max_length = len(text) + 1
        # sentencizer uses a rule based approach
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))
        #doc = nlp(text)
        #sentences = [sent.string.strip() for sent in doc.sents]
        sentences = sent_tokenize(text)
        print(len(sentences))
        with codecs.open(write_path, 'w', 'utf8') as outp:
            for sent in sentences:
                outp.write(sent)
                outp.write('\n')
            outp.close()

def preprocess_text8_gensim(read_path='./../data/text8', write_path='./../data/text8_sentences_gensim'):
    """
    >>> preprocess_text8_gensim()
    """
    sentences = word2vec.Text8Corpus(read_path)
    sentences = [sent for sent in sentences]
    words = set([item for sublist in sentences for item in sublist])
    print(len(sentences))
    with codecs.open(write_path, 'w', 'utf8') as outp:
        for sent in sentences:
            sentence = (' ').join(sent)
            outp.write(sentence)
            outp.write('\n')
        outp.close()


