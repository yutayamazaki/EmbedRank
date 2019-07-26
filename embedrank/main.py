from gensim.models.doc2vec import Doc2Vec

from embedrank import EmbedRank
from nlp_utils import tokenize

if __name__ == '__main__':
    model = Doc2Vec.load('jawiki_doc2vec_dmpv200d.model')
    embedrank = EmbedRank(model=model, tokenize=tokenize)

    text = """sample text Ja"""

    print(embedrank.extract_keyword(text))