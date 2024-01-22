# coding: utf-8
import sys
import numpy as np
from minhash import MinHash
from collections import defaultdict


class LSH(object):
    """ Locality Sensitive Hashing
    Band partitioned for approximate bucketing.
    """
    def __init__(self, b: int = 18, num_perm: int = 64):
        self.num_perm = num_perm
        self.b = b
        self.hashtables = [defaultdict(set) for i in range(self.b)]

        # band size
        self.r = self.num_perm // self.b
        # b-band partition intervals
        self.hashranges = [(i*self.r, (i+1)*self.r) for i in range(self.b)]

        self.keys = dict()

    def _H(self, hashvalues: np.ndarray):
        return bytes(hashvalues.byteswap().data)

    def insert(self, key: str, value: str, minhash: MinHash):
        if len(minhash.hashvalues) != self.num_perm:
            raise ValueError()

        Hs = [self._H(minhash.hashvalues[start:end])
              for start, end in self.hashranges]

        # Keep in the hashkey invt-table for similarity computation later on
        # Keep the value it self along with the hash for the value retrieval.
        for data in Hs:
            self.keys.setdefault(key, (minhash, value))

        # hashtable
        for H, hashtable in zip(Hs, self.hashtables):
            hashtable[H].add(key)

    def query(self, msg: str, semsim: bool=False):
        """Candidate bucket values and the actual pairwise similarity
        Args:
          msg : message of minhash to compare against the entire hash table
          semsim : if set True, filter out by similarity score, otherwise return all candidates.
                   (default: True)
        Returns:
          candidates : a list of tuples in the form (key, similarity[0:1])
        """
        minhash = MinHash(self.num_perm)
        minhash.update(msg)
        # TODO: simple threshold heuristic
        try:
            th = .5 if msg.count(' ') < 3 else .03
        except:
            th = .5
        th = 0

        if len(minhash.hashvalues) != self.num_perm:
            raise ValueError("Signature size don't match!")

        candidates = set()
        for (start, end), hashtable in zip(self.hashranges, self.hashtables):
            H = self._H(minhash.hashvalues[start:end])
            for key in hashtable.get(H, []):
                if semsim:
                    similarity = self.keys[key][0].semsim(minhash)
                    if similarity > th:
                        candidates.add((key, similarity, self.keys[key][1]))
                else:
                    candidates.add((key, self.keys[key][1]))

        if candidates:
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            return candidates[0]
        else:
            return []


if __name__ == '__main__':
    import codecs
    import time
    from tqdm import tqdm

    BANDWIDTH = 32
    N_PERMUTATION = BANDWIDTH*3

    lsh = LSH(b=BANDWIDTH, num_perm=N_PERMUTATION)

    print('Building LSH hash table')
    with codecs.open(sys.argv[1], 'r', 'utf-8') as fp:
        for query in tqdm(fp.read().split('\n'), ncols=80):
            try:
                if query:
                    hs = MinHash(num_perm=N_PERMUTATION)
                    hs.update(query)
                    lsh.insert(query, query, hs)
            except ValueError:
                pass

    while True:
        try:
            msg = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not msg:
            break
        p_t = time.time()
        print(lsh.query(msg, True))
        e_t = time.time()
        print('Took {} ms'.format(1000*(e_t-p_t)))

    print('Hashtable memory consumption (Bytes)')
    print(sys.getsizeof(lsh.hashtables))
    print('Inverted index  memory consumption (Bytes)')
    print(sys.getsizeof(lsh.keys))


