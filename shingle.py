# coding: utf-8
from itertools import combinations


def _get_partition(arr: list, k: int, n: int):
    """Generates all skip-n-grams with exactly k skips.
    Args:
      arr : array from which the skip-n-grams will be generated.
      k : the number of skips
      n : the size of n-grams
    Yields:
      n-gram with exactly k skips (as a tuple)
    """
    for i in range(len(arr) - n - k + 1):
        part = arr[i:i+n+k]

        if k == 0:
            yield tuple(part)
        else:
            for j in combinations(part[1:-1], n - 2):
                yield tuple([part[0]] + list(j) + [part[-1]])


def k_skip_n_grams(arr: list, k: int, n: int):
    """Generates all the k-skip-n-grams.
    Args:
      arr : array from which the skip-n-grams
          will be generated.
      n : the size of n-grams
      k : the maximum number of skips
    Yields:
      k-skip-n-gram (as a tuple)
    """
    if n == 0:
        return

    # Can't have skips in a unigram
    if n == 1:
        for e in arr:
            yield e
        return

    for i in range(0, min(len(arr) - n, k) + 1):
        yield from _get_partition(arr, i, n)


if __name__=='__main__':
    for gram in k_skip_n_grams('Howdy there how is life treating you?'.split(), 1, 3):
        print(gram)
