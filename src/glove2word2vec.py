from gensim.scripts.glove2word2vec import glove2word2vec
if __name__ == "__main__":
    glove2word2vec(glove_input_file="embeddings/glove.840B.300d/glove.840B.300d.txt", word2vec_output_file="embeddings/glove.840B.300d/glove.word2vec.txt")