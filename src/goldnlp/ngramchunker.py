import nltk

TRAIN = nltk.corpus.conll2000.chunked_sents("train.txt", chunk_types=["NP"])


class NGramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents=TRAIN, n=1):
        train_data = [
            [(t, c) for _, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents
        ]
        if n < 1 or n > 3:
            raise ValueError("n must be 1, 2, or 3")
        elif n == 1:
            self.tagger = nltk.UnigramTagger(train_data)
        elif n == 2:
            self.tagger = nltk.BigramTagger(train_data)
        elif n == 3:
            self.tagger = nltk.TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)
