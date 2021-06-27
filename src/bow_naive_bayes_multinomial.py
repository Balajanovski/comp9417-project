from preprocess import get_data

if __name__ == "__main__":
    X, y = get_data()
    for i in X.columns:
        print(i)