#
# Copyright 2014 Yasser Gonzalez Fernandez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Text classification based on character n-grams.
"""

import os
import codecs
import json
import itertools
import operator
import heapq


__version__ = '1.0.0'


class NGramProfile(object):
    """Character n-gram profile."""

    def __init__(self):
        """Initialize an empty profile."""
        self._ngrams = {}

    @classmethod
    def from_json(cls, file_path):
        """Load a profile previously saved in a JSON file."""
        profile = cls()
        with open(file_path, 'r') as fd:
            profile._ngrams = json.load(fd)
        return profile

    @classmethod
    def from_text(cls, text, ngram_sizes, profile_len, profile_offset):
        """Build a profile from a UTF-8 encoded string."""
        profile = cls()
        profile._count_ngrams(text, ngram_sizes)
        profile._normalize_ngram_freqs(ngram_sizes)
        profile._build_ngram_profile(profile_len, profile_offset)
        return profile

    @classmethod
    def from_file(cls, file_path, ngram_sizes, profile_len, profile_offset):
        """Build a profile from a UTF-8 encoded text file."""
        profile = cls.from_files((file_path, ), ngram_sizes,
                                 profile_len, profile_offset)
        return profile

    @classmethod
    def from_files(cls, file_paths, ngram_sizes, profile_len, profile_offset):
        """Build a profile from a list of UTF-8 encoded text files."""
        profile = cls()
        for file_path in file_paths:
            with codecs.open(file_path, 'r', 'utf-8') as fd:
                profile._count_ngrams(fd.read(), ngram_sizes)
        profile._normalize_ngram_freqs(ngram_sizes)
        profile._build_ngram_profile(profile_len, profile_offset)
        return profile

    @classmethod
    def from_dir(cls, dir_path, ngram_sizes, profile_len, profile_offset):
        """Build a profile from a directory containing UTF-8 encoded text files."""
        file_paths = []
        for dir_path, unused, file_names in os.walk(dir_path):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                file_paths.append(file_path)
        profile = cls.from_files(file_paths, ngram_sizes, profile_len, profile_offset)
        return profile

    def _count_ngrams(self, text, ngram_sizes):
        text = self.normalize(text)
        for ngram_size in ngram_sizes:
            slices = [itertools.islice(text, i, None) for i in xrange(ngram_size)]
            for ngram_tokens in itertools.izip(*slices):
                ngram = u''.join(ngram_tokens)
                self._ngrams[ngram] = self._ngrams.get(ngram, 0) + 1

    def _normalize_ngram_freqs(self, ngram_sizes):
        for ngram_size in ngram_sizes:
            ngram_count = 0.0
            for ngram in self._ngrams.iterkeys():
                if len(ngram) == ngram_size:
                    ngram_count += self._ngrams[ngram]
            for ngram in self._ngrams.iterkeys():
                if len(ngram) == ngram_size:
                    self._ngrams[ngram] = self._ngrams[ngram] / ngram_count

    def _build_ngram_profile(self, profile_len, profile_offset):
        top_ngrams = heapq.nlargest(profile_offset + profile_len,
                                    self._ngrams.iteritems(),
                                    key=operator.itemgetter(1))
        self._ngrams = dict(top_ngrams[profile_offset:])

    def __len__(self):
        """Number of n-grams in the profile."""
        return len(self._ngrams)

    def __iter__(self):
        """Return an iterator over the n-grams."""
        return self._ngrams.iterkeys()

    def __getitem__(self, ngram):
        """Return the n-gram frequency (zero if it does not appear)."""
        return self._ngrams.get(ngram, 0.0)

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

    def jaccard_dissimilarity(self, other):
        """One minus the Jaccard similarity coefficient.

        See e.g. http://en.wikipedia.org/wiki/Jaccard_index.
        """
        s = set(self)
        o = set(other)
        similarity = len(s & o) / float(len(s | o)) if s or o else 1
        return (1 - similarity)

    def cng_dissimilarity(self, other):
        """Common N-Grams (CNG) profile dissimilarity.

        See Vlado Keselj, Fuchun Peng, Nick Cercone, and Calvin Thomas (2003).
        "N-gram-based Author Profiles for Authorship Attribution". In Proceedings
        of the Conference Pacific Association for Computational Linguistics,
        PACLING'03, Nova Scotia, Canada, pp. 255-264.
        """
        dissimilarity = 0.0
        for ngram in set(self) | set(other):
            dissimilarity += (2 * (self[ngram] - other[ngram]) /
                              (self[ngram] + other[ngram])) ** 2
        return dissimilarity

    def out_of_place_dissimilarity(self, other):
        """Cavner-Trenkle out-of-place measure.

        See William B. Cavnar and John M. Trenkle (1994). "n-Gram-Based Text
        Categorization." In Proceedings of the 3rd Annual Symposium on Document
        Analysis and Information Retrieval, SDAIR'94, Las Vegas, US, pp. 161-175.
        """
        # Based on the implementation provided by the textcat R package
        # http://cran.r-project.org/web/packages/textcat/index.html.
        sorted_self = map(operator.itemgetter(0),
                          sorted(self._ngrams.iteritems(),
                                 key=operator.itemgetter(1), reverse=True))
        sorted_other = map(operator.itemgetter(0),
                           sorted(other._ngrams.iteritems(),
                                  key=operator.itemgetter(1), reverse=True))
        dissimilarity = 0
        for j in xrange(len(sorted_other)):
            if sorted_other[j] in self:
                i = sorted_self.index(sorted_other[j])
                dissimilarity += abs(i - j)
            else:
                dissimilarity += len(sorted_self)
        return dissimilarity
