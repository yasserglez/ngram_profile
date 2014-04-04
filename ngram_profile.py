"""Text classification based on n-grams."""

import json


class NGramProfile(object):
    """Base class for n-gram profiles.

    Subclass and override tokenize to provide an n-gram interpretation 
    (i.e. character n-grams, word n-grams, etc).
    """

    def __init__(self):
        """Initialize an empty profile."""
        self._ngrams = {} 

    @classmethod
    def from_json(cls, file_path):
        """Load a profile previously saved in a JSON file."""
        self = cls()
        with open(file_path, 'r') as fd:
            self._ngrams = json.load(fd)
        return self

    @classmethod
    def from_corpus(cls, corpus, ngram_size, profile_len, profile_offset):
        """Build a profile from a collection of text files."""

    def __len__(self):
        """Number of n-grams in the profile."""
        return len(self._ngrams)

    def __iter__(self):
        """Return an iterator over the n-grams."""
        return self._ngrams.iterkeys()

    def __getitem__(self, ngram):
        """Return the n-gram frequency (zero if it does not appear)."""
        return self._ngrams.get(ngram, 0)

    def __contains__(self, ngram):
        """Check if the profile contains an n-gram."""
        return ngram in self._ngrams

    def save_as_json(self, file_path):
        """Save the profile to a file in JSON format."""
        with open(file_path, 'w') as fd:
            json.dump(self._ngrams, fd)

    def normalize(self, text):
        """Text normalization (identity function by default)."""
        return text

    def tokenize(self, text):
        """Split text into tokens.

        Not implemented by default. Override in a subclass to provide an n-gram
        interpretation (i.e. character n-grams, word n-grams, etc).
        """
        raise NotImplementedError()

    def jaccard_dissimilarity(self, other):
        """One minus the Jaccard similarity coefficient.

        See e.g. http://en.wikipedia.org/wiki/Jaccard_index.
        """

    def cng_dissimilarity(self, other):
        """Common N-Grams (CNG) profile dissimilarity.

        See Vlado Keselj, Fuchun Peng, Nick Cercone, and Calvin Thomas (2003). 
        "N-gram-based Author Profiles for Authorship Attribution". In Proceedings
        of the Conference Pacific Association for Computational Linguistics, 
        PACLING'03, Nova Scotia, Canada, pp. 255-264.
        """

    def out_of_place_dissimilarity(self, other):
        """Cavner-Trenkle out-of-place measure.

        See William B. Cavnar and John M. Trenkle (1994). "n-Gram-Based Text 
        Categorization." In Proceedings of the 3rd Annual Symposium on Document
        Analysis and Information Retrieval, SDAIR'94, Las Vegas, US, pp. 161-175.
        """


class CharNGramProfile(NGramProfile):
    """Character-based n-gram profile."""

    def normalize(self, text):
        """Text normalization (identity function by default)."""
        return text

    def tokenize(self, text):
        """Split text into characters."""
