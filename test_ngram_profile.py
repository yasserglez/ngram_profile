#
# Copyright 2014 Yasser Gonzalez Fernandez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
import json
import unittest
import codecs

from ngram_profile import NGramProfile


class TestNGramProfile(unittest.TestCase):

    def test_init(self):
        profile = NGramProfile()
        self.assertEqual(len(profile), 0)

    def test_json_roundtrip(self):
        json_profile = '{"a": 0.5, "b": 0.3, "c": 0.2}'
        tmp_file = 'test_json_roundtrip.json'
        with codecs.open(tmp_file, 'w', 'utf-8') as fd:
            fd.write(json_profile)
        profile = NGramProfile.from_json(tmp_file)
        os.remove(tmp_file)
        self.assertEqual(len(profile), 3)
        self.assertEqual(profile[u'a'], 0.5)
        self.assertEqual(profile[u'b'], 0.3)
        self.assertEqual(profile[u'c'], 0.2)
        profile.save_as_json(tmp_file)
        with codecs.open(tmp_file, 'r', 'utf-8') as fd:
            self.assertEqual(json.load(fd), json.loads(json_profile))
        os.remove(tmp_file)

    def test_normalize(self):
        text = u'abc'
        profile = NGramProfile()
        normalized_text = profile.normalize(text)
        self.assertTrue(isinstance(normalized_text, unicode))
        self.assertEqual(normalized_text, text)

    def test_1gram(self):
        text = u'abcaab'
        ngram_sizes = (1, )
        profile_len = 3
        profile = NGramProfile.from_text(text, ngram_sizes, profile_len, 0)
        self.assertEqual(len(profile), profile_len)
        self.assertEqual(profile[u'a'], 0.5)
        self.assertAlmostEqual(profile[u'b'], 0.33, delta=0.01)
        self.assertAlmostEqual(profile[u'c'], 0.16, delta=0.01)

    def test_2gram(self):
        text = u'abcaab'
        ngram_sizes = (2, )
        profile_len = 4
        profile = NGramProfile.from_text(text, ngram_sizes, profile_len, 0)
        self.assertEqual(len(profile), profile_len)
        self.assertEqual(profile[u'ab'], 0.4)
        self.assertEqual(profile[u'bc'], 0.2)
        self.assertEqual(profile[u'ca'], 0.2)
        self.assertEqual(profile[u'aa'], 0.2)

    def test_1gram_and_2gram_with_offset(self):
        text = u'abcaab'
        ngram_sizes = (1, 2)
        profile_len = 4
        profile_offset = 2
        profile = NGramProfile.from_text(text, ngram_sizes, profile_len, profile_offset)
        self.assertEqual(len(profile), profile_len)
        self.assertAlmostEqual(profile[u'b'], 0.33, delta=0.01)
        self.assertEqual(profile[u'bc'], 0.2)
        self.assertEqual(profile[u'ca'], 0.2)
        self.assertEqual(profile[u'aa'], 0.2)

    def test_jaccard_dissimilarity(self):
        test_cases = (
            (u'', u'', 0),
            (u'abc', u'', 1),
            (u'', u'fgh', 1),
            (u'abc', u'fgh', 1),
            (u'abcde', u'defgh', 0.75),
            (u'de', u'de', 0),
        )
        ngram_sizes = (1, )
        profile_len = 6
        profile_offset = 0
        for test_case in test_cases:
            profile1 = NGramProfile.from_text(test_case[0],
                    ngram_sizes, profile_len, profile_offset)
            profile2 = NGramProfile.from_text(test_case[1],
                    ngram_sizes, profile_len, profile_offset)
            dissimilarity = profile1.jaccard_dissimilarity(profile2)
            self.assertEqual(dissimilarity, test_case[2])

    def test_cng_dissimilarity(self):
        ngram_sizes = (1, )
        profile_len = 2
        profile_offset = 0
        profile1 = NGramProfile.from_text(u'abb',
                ngram_sizes, profile_len, profile_offset)
        profile2 = NGramProfile.from_text(u'aac',
                ngram_sizes, profile_len, profile_offset)
        dissimilarity = profile1.cng_dissimilarity(profile2)
        self.assertAlmostEqual(dissimilarity, 8.44, delta=0.01)

    def test_out_of_place_dissimilarity(self):
        tmp_file = 'test_out_of_place_dissimilarity.json'
        json_profile = '{"TH": 6, "ER": 5, "ON": 4, "LE": 3, "ING": 2, "AND": 1}'
        with codecs.open(tmp_file, 'w', 'utf-8') as fd:
            fd.write(json_profile)
        profile1 = NGramProfile.from_json(tmp_file)
        json_profile = '{"TH": 6, "ING": 5, "ON": 4, "ER": 3, "AND": 2, "ED": 1}'
        with codecs.open(tmp_file, 'w', 'utf-8') as fd:
            fd.write(json_profile)
        profile2 = NGramProfile.from_json(tmp_file)
        os.remove(tmp_file)
        dissimilarity = profile1.out_of_place_dissimilarity(profile2)
        self.assertEqual(dissimilarity, 12)


if __name__ == '__main__':
    unittest.main()
