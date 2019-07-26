import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbedRank:
    """ Implementation of EmbedRank and EmbedRank++ introduced
        https://arxiv.org/abs/1801.04470.

    Parameters
    ----------
    tagger: MeCab.Tagger
        Tokenizer for Japanese.
    
    doc2vec: gensim.models.Doc2Vec

    word2vec: gensim.models.Word2Vec
    """

    def __init__(self, tagger, doc2vec, word2vec, alpha=0.8):
        self.tagger = tagger
        self.d2v = doc2vec
        self.w2v = word2vec
        self.alpha = alpha

    def extract_keywords(self, doc: str, top_k=10, mmr=True):
        sentense_embed = self._calc_sentence_embedding(doc)
        candicates = self._get_keyphrase_candidates(doc)
        word_embed = self._calc_word_embedding(candicates)

        if mmr:
            similarities = self._mmr(sentense_embed, word_embed)
        else:
            similarities = []
            for w, w_vec in word_embed.items():
                sim_score = cosine_similarity(w_vec.reshape(1, -1), sentense_embed.reshape(1, -1))
                similarities.append((w, sim_score))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
    
    def _mmr(self, sentense_embed, word_embed):
        candidates = []
        for c_i, c_i_embed in word_embed.items():
            sim_1 = cosine_similarity(c_i_embed.reshape(1, -1), sentense_embed.reshape(1, -1))

            sim_2 = []
            for c_j, c_j_embed in word_embed.items():
                if c_i != c_j:
                    sim_2.append(cosine_similarity(c_i_embed.reshape(1, -1), c_j_embed.reshape(1, -1)))

            candidates.append((c_i, self.alpha*sim_1 - (1-self.alpha)*np.max(sim_2)))

        return candidates

    def _get_keyphrase_candidates(self, doc: str) -> list:
        return self._tokenize(doc)

    def _calc_sentence_embedding(self, doc: str) -> np.ndarray:
        words = self._tokenize(doc)
        sent_embed = self.d2v.infer_vector(words)
        return sent_embed

    def _calc_word_embedding_(self, candidates: list) -> dict:
        vectors = {}
        for c in candidates:
            try:
                v = self.w2v.wv[c]
            except Exception as e:
                continue
            vectors[c] = v
        return vectors

    def _calc_word_embedding(self, candidates: list) -> dict:
        vectors = {}
        for c in candidates:
            try:
                v = self.d2v.infer_vector([c])
            except Exception as e:
                continue
            vectors[c] = v
        return vectors

    def _tokenize(self, doc: str, attr=['名詞', '形容詞']) -> list:
        result = []
        node = self.tagger.parseToNode(doc)
        while node:
            if node.feature.split(',')[6] == '*':
                word = node.surface
            else:
                word = node.feature.split(',')[6]

            part = node.feature.split(',')[0]

            if part in attr:
                result.append(word)
            node = node.next
        return result