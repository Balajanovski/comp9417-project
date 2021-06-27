import util
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from time import time
import numpy as np
from scipy.sparse import lil_matrix

st = time()

def words_to_vec_multinomial(text, word_map):
    ret = lil_matrix((1,len(word_map)))
    if isinstance(text, float):
        print("float",text)
    for word in text.split(" "):
        x = word_map.get(word)
        if x:
            ret[0,x] += 1
    return ret

if __name__ == "__main__":
    word_map = util.load_word_map()
    X, y = util.load_train(word_map, words_to_vec_multinomial)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
    model = MultinomialNB()
    model.fit(X_train,y_train)

    total = X_test.shape[0]
    y_pred = model.predict(X_test)
    count = np.sum(y_pred == y_test)
    print(f"Accuracy: {count/total}")

print(f"time: {time()-st}s")