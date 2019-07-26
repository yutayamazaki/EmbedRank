from gensim.models import Doc2Vec, Word2Vec
import MeCab

from embedrank import EmbedRank


if __name__ == '__main__':
    doc2vec = Doc2Vec.load('jawiki_doc2vec_dmpv200d.model')
    word2vec = Word2Vec.load('jawiki_word2vec_200d.model')
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    embed_rank = EmbedRank(tagger, doc2vec, word2vec)
    text = 'EmbedRankという，単一文書からの教師なしキーフレーズ抽出'

    keywords = embed_rank.extract_keywords(text, mmr=True)
    print(keywords)

    keywords = embed_rank.extract_keywords(text, mmr=False)
    print(keywords)