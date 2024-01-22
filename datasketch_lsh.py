import re
import numpy as np
from datasketch import MinHash, MinHashLSH

# Function to preprocess text into words
def preprocess_text(text):
    return re.findall(r'\b\w+\b', text.lower())

# Function to create MinHash signatures for a set of words
def create_minhash_signature(words, num_perm=128):
    minhash = MinHash(num_perm=num_perm)
    for word in words:
        minhash.update(word.encode('utf-8'))
    return minhash

# Function to identify and remove fuzzy duplicates using LSH
def remove_fuzzy_duplicates(input_file, output_file, threshold=0.7):
    # Initialize LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)

    # Read the large file and process each line
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            words = preprocess_text(line)
            minhash_signature = create_minhash_signature(words)
            
            # Query LSH index to check for duplicates
            duplicates = lsh.query(minhash_signature)

            # If no duplicates found, add the current signature to the index
            if not duplicates:
                lsh.insert(line, minhash_signature)

    # Write the non-duplicate lines to the output file
    with open(output_file, 'w', encoding='utf-8') as output_file:
        with open(input_file, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                words = preprocess_text(line)
                minhash_signature = create_minhash_signature(words)

                # Check if the line is not a duplicate
                if line in lsh.query(minhash_signature):
                    output_file.write(line)

# Example usage
remove_fuzzy_duplicates('input.txt', 'output_deduped.txt')

