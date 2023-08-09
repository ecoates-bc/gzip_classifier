from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import gzip
import numpy as np
import cppimport.import_hook
import ncd


def compute_ncd(a, b):
    a_bytes = a.toarray().tobytes()
    b_bytes = b.toarray().tobytes()

    c_ab = len(gzip.compress(a_bytes + b_bytes))
    c_a = len(gzip.compress(a_bytes))
    c_b = len(gzip.compress(b_bytes))
    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)


def main():
    ncd.test()
    ag_news_data = load_dataset("ag_news")
    train_ds, test_ds = ag_news_data["train"], ag_news_data["test"]

    N_TRAINING_EXAMPLES = 100
    N_TEST_EXAMPLES = 500

    train_ds = train_ds.shuffle(seed=42)
    test_ds = test_ds.shuffle(seed=42)

    vectorizer = CountVectorizer()

    x_examples = [train_ds[i]["text"] for i in range(N_TRAINING_EXAMPLES)]
    X = vectorizer.fit_transform(x_examples)

    y = np.asarray(train_ds["label"][:N_TRAINING_EXAMPLES])

    x_test_examples = [test_ds[i]["text"] for i in range(N_TEST_EXAMPLES)]
    X_test = vectorizer.transform(x_test_examples)
    y_test = np.asarray(test_ds["label"][:N_TEST_EXAMPLES])

    print("Initializing model...")
    model = KNeighborsClassifier(
        n_neighbors=5,
        metric=compute_ncd,
    )

    print("Fitting model...")
    model.fit(X, y)

    print("Training accuracy:", model.score(X, y))
    print("Test accuracy:", model.score(X_test, y_test))

    print("Complete!")


if __name__ == "__main__":
    main()