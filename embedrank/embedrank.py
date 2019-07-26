import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

from nlp_utils import extract_keyphrase_candidates, tokenize


class EmbedRank:

    def __init__(self, model, tokenize, l=0.80):
        self.model = model
        self.tokenize = tokenize
        self.l = l

    def extract_keyword(self, text, top_k=5):
        self.top_k = int(top_k)

        phrases, phrase_embeds, doc_embed = self._calc_embeddings(text)
        if not phrases:
            return []
        phrase_indices, doc_similarity = self._mmr(phrases, phrase_embeds, doc_embed)

        output = []
        for idx in phrase_indices:
            output.append((phrases[idx], doc_similarity[idx][0]))
        return output

    def _calc_embeddings(self, document):
        doc_embed = self.model.infer_vector(self.tokenize(document))

        phrases = []
        phrase_embeds = []
        for candidate_tokens in extract_keyphrase_candidates(document):
            candidate_text = ''.join(candidate_tokens)
            phrases.append(candidate_text)
            phrase_embeds.append(self.model.infer_vector(candidate_tokens))
        phrase_embeds = np.array(phrase_embeds)

        if len(phrases) == 0:
            return [], [], doc_embed
        if len(phrases) < self.top_k:
            self.top_k = len(phrases)

        return phrases, phrase_embeds, doc_embed

    def _mmr(self, phrases, phrase_embeds, doc_embed):

        doc_similarity = cosine_similarity(phrase_embeds, doc_embed.reshape(1, -1))
        phrase_similarity_matrix = cosine_similarity(phrase_embeds)

        unselected = list(range(len(phrases)))
        select_idx = np.argmax(doc_similarity)

        selected = [select_idx]
        unselected.remove(select_idx)

        for _ in range(self.top_k - 1):
            mmr_distance_to_doc = doc_similarity[unselected, :]
            mmr_distance_between_phrases = np.max(phrase_similarity_matrix[unselected][:, selected], axis=1)

            mmr = self.l * mmr_distance_to_doc - (1 - self.l) * mmr_distance_between_phrases.reshape(-1, 1)
            mmr_idx = unselected[np.argmax(mmr)]

            selected.append(mmr_idx)
            unselected.remove(mmr_idx)

        return selected, doc_similarity