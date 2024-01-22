# coding: utf-8
import binascii
import re
import sys

import numpy as np

from shingle import k_skip_n_grams

# [Mersenne Prime number](https://en.wikipedia.org/wiki/Mersenne_prime)
PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1
HASH_RANGE = 1 << 32


def normalize(v: np.ndarray):
    norm = np.linalg.norm(v, ord=2)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


class MinHash(object):
    """MinHash object
    Initiate with a numpy array permutation,
    creatng a min-hash vector for the given string.
    """
    def __init__(self, num_perm=64, seed=1):
        self.num_perm = num_perm
        if HASH_RANGE < num_perm:
            raise ValueError()

        self.seed = seed
        self.hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH

        generator = np.random.RandomState(self.seed)
        self.permutations = np.array(
            [
                (
                    generator.randint(1, PRIME, dtype=np.uint64),
                    generator.randint(0, PRIME, dtype=np.uint64),
                )
                for _ in range(num_perm)
            ],
            dtype=np.uint64,
        ).T

    def _hashfunc(self, b: bytes):
        """Create hash base value given a bytes array
        Returns:
          Masked unsigned integer value
        """
        return binascii.crc32(b) & MAX_HASH

    def update(self, s: str):
        """Compute min-hash values of the given string"""
        shingles = self.shingling(s)
        for shingle in shingles:
            hv = self._hashfunc(shingle)
            a, b = self.permutations
            phv = np.bitwise_and((a * hv + b) % PRIME, np.uint64(MAX_HASH))
            self.hashvalues = np.minimum(phv, self.hashvalues)

        self.vector = self.vectorize()

    def shingling(self, s: str, n: int = 3):
        """Creates pieces skip-grams for hash creation
        Args:
          s     : string to create shingles list
          n [3] : integer n-gram size
        Returns:
          A list of shingles encoded as utf-8 binary ascii
        """
        s = re.sub(r"([^\w ])+", "", s.lower())
        # word level skip-bigrams + char level skip-trigrams
        res = ["^".join(grams) for grams in k_skip_n_grams(s.split(), 1, 2)]
        res.extend(["^".join(grams) for grams in k_skip_n_grams(s, 1, 3)])
        return [r.encode("utf-8") for r in res]

    def vectorize(self):
        """Compute an iterative binary hashing on hashvalues
        Returns:
          A float32 vector of 0, 1 in num_perm**2 dimension
        """
        vec = []
        for i in range(self.num_perm):
            for j in range(self.num_perm):
                a = self.permutations[0][j]
                b = self.permutations[1][j]
                vec.append((self.hashvalues[i] / b) % 2)
        return normalize(np.asarray(vec))

    def cossim(self, other):
        """Compute cosine similarity of vectorized forms
        Returns:
          Cosine-similarity in numpy float
        """
        return np.dot(self.vector, other.vector)

    def jaccard(self, other):
        """Compute approximated jaccard set-similarity using minhash
        Args:
          other : MinHash instance to compare against
        Returns:
          Jaccard similarity in numpy.float32
        """
        if other.seed != self.seed:
            raise ValueError()

        if len(self.hashvalues) != len(other.hashvalues):
            raise ValueError()

        return np.float32(
            np.count_nonzero(self.hashvalues == other.hashvalues)
        ) / np.float32(len(self.hashvalues))

    def semsim(self, other):
        """Cos-sim weighted jaccard similarity
        Returns:
          Exponential weighed similarity
        """
        jsim = self.jaccard(other)
        csim = self.cossim(other)
        return jsim * np.sin(np.pi * csim / 2)


def argmax(arr):
    max_i = 0
    max_v = arr[0]
    for i, e in enumerate(arr):
        if e > max_v:
            max_i = i
            max_v = e
    return max_i


if __name__ == "__main__":
    data = [
        "What should we grab for lunch?",
        "What about tomorrow?",
        "Can you hear me?",
        "I am so exhuasted",
    ]
    index = []
    for d in data:
        # create an insert minhash values in the index array
        m = MinHash()
        m.update(d)
        index.append(m)

    print(f"Query against : {data}")
    while True:
        # query against each entry in the index array
        try:
            msg = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not msg:
            break

        mh = MinHash()
        mh.update(msg)

        similarities = []
        for h in index:
            similarities.append(h.semsim(mh))
        i = argmax(similarities)
        print(similarities)
        print(f"{data[i]} : {100*similarities[i]:.3f}%")
