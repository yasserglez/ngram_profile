"""Text classification based on n-grams."""


class NGramProfile(object):
    """Base class for n-gram profiles.

    Not intended to be instantiated. Subclass and override tokenize to provide 
    an n-gram interpretation (i.e. character n-grams, word n-grams, etc).
    """

    def __init__(self):
        """Initialize an empty profile."""

    @classmethod
    def from_json(cls, file_path):
        """Load a profile previously saved in a JSON file."""

    @classmethod
    def from_corpus(cls, corpus, ngram_size, profile_len, profile_offset):
        """Build a profile from a collection of text files."""

    def __len__(self):
        """Number of n-grams in the profile."""

    def __iter__(self):
        """Return an iterator over the n-grams."""

    def __getitem__(self, ngram):
        """Return the n-gram frequency (zero if it does not appear)."""

    def __contains__(self, ngram):
        """Check if the profile contains an n-gram."""

    def save_as_json(self, file_path):
        """Save the profile to a file in JSON format."""

    def normalize(self, text):
        """Text normalization (identity function by default)."""

    def tokenize(self, text):
        """Split text into tokens.

        Not implemented by default. Override in a subclass to provide an n-gram
        interpretation (i.e. character n-grams, word n-grams, etc).
        """

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
