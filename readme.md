# Text classification with GZIP

This is a model implementation inspired by "A Parameter-Free Classification Method With Compressors" (https://aclanthology.org/2023.findings-acl.426/) which appeared in the findings of this year's ACL conference. They were able to get classification accuracy results comparable to a pretrained BERT classifier by using a combination of Gzip and a KNN classifier.

Here's the general workflow of the code in `gzip_classifier.py`:

- Download the AG news dataset from HuggingFace (a collection of short excerpts from newspapers, accompanied by their topic)
- Precompute a distance matrix with each example pair in the training set. This is done in parallel, using the Normalized Compression Distance metric (described in the paper). Each text pair is gzipped in order to calculate NCD.
- Pass the distance matrix and labels to a sklearn KNN classifier, and fit.

It's quite simple! However, the big drawbacks for me are speed and memory. Since KNNs are non-parametric, the model size scales in n^2. Also, the precompute step is n^2. Once the training set got beyond 5000 examples in size, the model became too impractical to run on just one computer. Perhaps this could be sped up by digging "under the hood" of sklearn's KNN classifier somehow.

## Running the model
It's implemented in Python and can be run from the command line:
```
pip install --break-system-packages -r requirements.txt
python gzip_classifier.py
```