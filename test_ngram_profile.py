# -*- coding: utf-8 -*-

import os
import json
import unittest

from ngram_profile import NGramProfile, CharNGramProfile


class TestNGramProfile(unittest.TestCase):

    def test_init(self):
        profile = NGramProfile()
        self.assertEqual(len(profile), 0)

    def test_json_roundtrip(self):
        json_profile = '{"a": 0.5, "b": 0.3, "c": 0.2}'
        tmp_file = 'test_json_roundtrip.json'
        with open(tmp_file, 'w') as fd:
            fd.write(json_profile)
        profile = NGramProfile.from_json(tmp_file)
        os.remove(tmp_file)
        self.assertEqual(len(profile), 3)
        self.assertEqual(profile[u'a'], 0.5)
        self.assertEqual(profile[u'b'], 0.3)
        self.assertEqual(profile[u'c'], 0.2)
        profile.save_as_json(tmp_file)
        with open(tmp_file, 'r') as fd:
            self.assertEqual(json.load(fd), json.loads(json_profile))
        os.remove(tmp_file) 

    def test_normalize(self):
        profile = NGramProfile()
        x = u'abc'
        y = profile.normalize(x)
        self.assertTrue(isinstance(y, unicode))
        self.assertEqual(x, y)

    def test_tokenize(self):
        profile = NGramProfile()
        self.assertRaises(NotImplementedError, profile.tokenize, u'')


if __name__ == '__main__':
    unittest.main()
