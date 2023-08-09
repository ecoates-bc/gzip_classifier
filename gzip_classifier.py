from datasets import load_dataset
import numpy as np
import gzip
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from multiprocessing import Pool
import ncd
from numba import jit


def create_bytestring(example, i):
    to_bytes = bytes(example["text"], "utf-8")
    example["bytes"] = to_bytes
    return example


def compute_ncd_python(a, b):
    """Compute the normalized compression distance. a and b are byte strings.
    """
    c_ab = len(gzip.compress(a + b))
    c_a = len(gzip.compress(a))
    c_b = len(gzip.compress(b))
    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)


def compute_ncd(a, b):
    return ncd.compute_ncd(a, b)


def update_matrix(x, i, j, result):
    x[i, j] = result


def compute_and_return_ncd(starmap_args):
    a, b, i, j = starmap_args
    ncd = compute_ncd_python(a, b)
    return i, j, ncd


def precompute_X(bytestrings, n):
    X = np.zeros((n, n))
    with Pool() as pool:
        progress = tqdm(total=(n**2)*2, desc="Precomputing X...")
        starmap_args = []
        for i in range(n):
            for j in range(n):
                a = bytestrings[i]
                b = bytestrings[j]
                starmap_args.append((a, b, i, j))
                progress.update(1)
        
        for result in pool.imap(compute_and_return_ncd, starmap_args, chunksize=250):
            i, j, ncd = result
            update_matrix(X, i, j, ncd)
            progress.update(1)
    
    return X


def precompute_X_test(bytestrings, bytestrings_test, n, n_test):
    X_test = np.zeros((n_test, n))
    with Pool() as pool:
        progress = tqdm(total=n*n_test, desc="Precomputing X_test...")
        starmap_args = []
        for i in range(n_test):
            for j in range(n):
                progress.update(1)
                a = bytestrings_test[i]
                b = bytestrings[j]
                starmap_args.append((a, b, i, j))

        for result in pool.imap(compute_and_return_ncd, starmap_args, chunksize=250):
            i, j, ncd = result
            update_matrix(X_test, i, j, ncd)
            progress.update(1)

    return X_test


def main():
    ag_news_data = load_dataset("ag_news")
    train_ds_raw, test_ds_raw = ag_news_data["train"], ag_news_data["test"]

    N_TRAINING_EXAMPLES = 5000
    N_TEST_EXAMPLES = 500

    train_ds = train_ds_raw.map(create_bytestring, with_indices=True).shuffle(seed=42)
    test_ds = test_ds_raw.map(create_bytestring, with_indices=True).shuffle(seed=42)
    
    # Create X by precomputing pairwise distances
    n = N_TRAINING_EXAMPLES
    bytestrings = [train_ds[i]["bytes"] for i in range(n)]
    X = precompute_X(bytestrings, n)
    y = np.asarray(train_ds["label"][:n])

    # Do the same for X_test
    n_test = N_TEST_EXAMPLES
    bytestrings_test = [test_ds[i]["bytes"] for i in range(n_test)]
    X_test = precompute_X_test(bytestrings, bytestrings_test, n, n_test)
    y_test = np.asarray(test_ds["label"][:n_test])

    print("Initializing model...")
    model = KNeighborsClassifier(
        n_neighbors=5,
        metric="precomputed",
    )

    print("Fitting model...")
    model.fit(X, y)

    print("Training acuraccy:", model.score(X, y))
    print("Test accuracy:", model.score(X_test, y_test))

    print("Complete!")


if __name__ == "__main__":
    main()