# -*- coding: utf-8 -*-

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

    def test_from_text_1gram(self):
        text = u'abcaab'
        ngram_sizes = (1, )
        profile_len = 3
        profile = NGramProfile.from_text(text, ngram_sizes, profile_len, 0)
        self.assertEqual(len(profile), profile_len)
        self.assertEqual(profile[u'a'], 0.5)
        self.assertAlmostEqual(profile[u'b'], 0.33, delta=0.01)
        self.assertAlmostEqual(profile[u'c'], 0.16, delta=0.01)

    def test_from_text_2gram(self):
        text = u'abcaab'
        ngram_sizes = (2, )
        profile_len = 4
        profile = NGramProfile.from_text(text, ngram_sizes, profile_len, 0)
        self.assertEqual(len(profile), profile_len)
        self.assertEqual(profile[u'ab'], 0.4)
        self.assertEqual(profile[u'bc'], 0.2)
        self.assertEqual(profile[u'ca'], 0.2)
        self.assertEqual(profile[u'aa'], 0.2)

    def test_from_text_1gram_2gram_with_offset(self):
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


if __name__ == '__main__':
    unittest.main()
