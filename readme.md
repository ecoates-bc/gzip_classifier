# Text classification with GZIP

This is a model implementation inspired by "A Parameter-Free Classification Method With Compressors" (https://aclanthology.org/2023.findings-acl.426/) which appeared in the findings of this year's ACL conference. They were able to get classification accuracy results comparable to a pretrained BERT classifier by using a combination of Gzip and a KNN classifier.

Here's the general workflow of the code in `gzip_classifier.py`:

- Download the AG news dataset from HuggingFace (a collection of short excerpts from newspapers, accompanied by their topic)
- Precompute a distance matrix with each example pair in the training set. This is done in parallel, using the Normalized Compression Distance metric (described in the paper). Each text pair is gzipped in order to calculate NCD.
- Pass the distance matrix and labels to a sklearn KNN classifier, and fit.

It's quite simple! And it works reasonably well - the best test accuracy I found was 0.75 (with chance being 0.25). It's not as good as the paper's best result for AG news, and I think it's because this model is training on a much smaller slice of the entire database. Since KNNs are non-parametric, the model size scales in n^2. Also, the precompute step is n^2 for time. Once the training set got beyond 5000 examples in size, the model became too impractical to run on just one computer - either the precompute step takes impractically long, or the required memory goes past 100 GB. Perhaps the model could made less expensive by digging "under the hood" of sklearn's KNN classifier somehow.

## Running the model
It's implemented in Python and can be run from the command line:
```
pip install --break-system-packages -r requirements.txt
python gzip_classifier.py
```
