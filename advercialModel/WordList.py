import os
import json
import nltk
from collections import Counter
from typing import List

def generate_frequent_word_list(
    language: str = 'english',
    output_path: str=None,
    limit: int = 20000,
    min_length: int = 5,
    max_length: int = 20,
    exclude_proper_nouns: bool = True,
    exclude_punctuation: bool = True,
    lowercase: bool = True,
    verbose: bool = True
) -> List[str]:
    """
    Generates a word list based on word frequency using NLTK's corpora.

    Parameters:
        output_path (str): Path to save the generated word list.
        language (str): Language of the word list (e.g., 'english', 'spanish', 'italian', 'basque').
        limit (int): Maximum number of words to include in the list.
        min_length (int): Minimum length of words to include.
        max_length (int): Maximum length of words to include.
        exclude_proper_nouns (bool): Whether to exclude proper nouns.
        exclude_punctuation (bool): Whether to exclude words with punctuation.
        lowercase (bool): Whether to convert words to lowercase.
        verbose (bool): Whether to print progress.

    Returns:
        List[str]: A list of filtered frequent words.
    """
    # Map languages to their corresponding corpus names and file IDs in NLTK

    if output_path is None:
        output_path = f"{language}_frequent_word_list.json"
    corpus_map = {
        'english': {'corpus': 'brown', 'fileid': None},
        'spanish': {'corpus': 'cess_esp', 'fileid': None},
        'italian': {'corpus': 'udhr', 'fileid': 'Italian-Latin1'},
        'basque': {'corpus': 'conll2007', 'fileid': 'eus.train'}
    }

    if language not in corpus_map:
        raise ValueError(f"Unsupported language: {language}. Choose from {list(corpus_map.keys())}.")

    corpus_info = corpus_map[language]
    corpus_name = corpus_info['corpus']
    fileid = corpus_info['fileid']

    # Check if the word list already exists
    if os.path.isfile(output_path):
        if verbose:
            print(f"Loading existing word list from {output_path}...")
        with open(output_path, 'r', encoding='utf-8') as f:
            word_list = json.load(f)
        if verbose:
            print(f"Loaded {len(word_list)} words.")
        return word_list

    # Download required NLTK corpora if not already present
    try:
        nltk.data.find(f'corpora/{corpus_name}')
    except LookupError:
        if verbose:
            print(f"Downloading NLTK '{corpus_name}' corpus for {language} frequency analysis...")
        nltk.download(corpus_name)

    # Retrieve words from the selected corpus
    try:
        if corpus_name == 'conll2007':
            # For Basque, using 'eus.train' from 'conll2007' corpus
            words_corpus = nltk.corpus.conll2007.words(fileid)
        elif corpus_name == 'udhr':
            # For Italian, using 'Italian-Latin1' from 'udhr' corpus
            words_corpus = nltk.corpus.udhr.words(fileid)
        else:
            # Dynamically access the corpus using getattr
            words_corpus = getattr(nltk.corpus, corpus_name).words()
    except AttributeError:
        raise AttributeError(f"The corpus '{corpus_name}' is not available in NLTK.")
    except LookupError:
        raise LookupError(f"The file ID '{fileid}' was not found in the '{corpus_name}' corpus.")

    if verbose:
        print(f"Total words in {language} corpus: {len(words_corpus)}")

    # Compute word frequencies
    word_freq = Counter([word.lower() for word in words_corpus])

    # Sort words by frequency
    sorted_words = [word for word, freq in word_freq.most_common()]

    # Initialize an empty set to avoid duplicates
    filtered_words = set()

    for word in sorted_words:
        # Apply length filters
        if len(word) < min_length or len(word) > max_length:
            continue

        # Exclude proper nouns by checking capitalization
        if exclude_proper_nouns and word[0].isupper():
            continue

        # Exclude words with punctuation
        if exclude_punctuation and not word.isalpha():
            continue

        # Convert to lowercase if specified
        if lowercase:
            word = word.lower()

        filtered_words.add(word)

        # Stop if the limit is reached
        if len(filtered_words) >= limit:
            break

    # Convert the set to a sorted list
    word_list = sorted(list(filtered_words))
    if verbose:
        print(f"Filtered frequent words count: {len(word_list)}")

    # Save the word list to the specified JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(word_list, f, ensure_ascii=False, indent=4)
    if verbose:
        print(f"Word list saved to {output_path}")

    return word_list

# Main function for generating and displaying the top 10 frequent words for each language
if __name__ == "__main__":
    languages = ['english', 'spanish', 'italian', 'basque']
    for language in languages:
        output_path = f"{language}_frequent_word_list.json"
        print(f"\nGenerating frequent word list for {language}...")
        try:
            word_list = generate_frequent_word_list(output_path=output_path, language=language, verbose=True)
            # Print the first 10 words in the generated list
            print(f"First 10 frequent words in {language}:")
            print(word_list[:10])
        except Exception as e:
            print(f"An error occurred while processing {language}: {e}")
