# -*- coding: utf-8 -*-

import os
import json
import unittest

import ngram_profile


class CommonNGramProfileTests(object):

    profileClass = None

    def test_init(self):
        profile = self.profileClass()
        self.assertEqual(len(profile), 0)

    def test_json_roundtrip(self):
        json_profile = '{"a": 0.5, "b": 0.3, "c": 0.2}'
        tmp_file = 'test_json_roundtrip.json'
        with open(tmp_file, 'w') as fd:
            fd.write(json_profile)
        profile = self.profileClass.from_json(tmp_file)
        os.remove(tmp_file)
        self.assertEqual(len(profile), 3)
        self.assertEqual(profile[u'a'], 0.5)
        self.assertEqual(profile[u'b'], 0.3)
        self.assertEqual(profile[u'c'], 0.2)
        profile.save_as_json(tmp_file)
        with open(tmp_file, 'r') as fd:
            self.assertEqual(json.load(fd), json.loads(json_profile))
        os.remove(tmp_file) 

    def test_normalize_unicode_output(self):
        profile = self.profileClass()
        normalized = profile.normalize(u'abc')
        self.assertTrue(isinstance(normalized, unicode))


class TestNGramProfile(CommonNGramProfileTests, unittest.TestCase):

    profileClass = ngram_profile.NGramProfile

    def test_normalize(self):
        profile = self.profileClass()
        x = u'abc'
        y = profile.normalize(x)
        self.assertEqual(x, y)

    def test_tokenize(self):
        profile = ngram_profile.NGramProfile()
        self.assertRaises(NotImplementedError, profile.tokenize, u'')


class TestCharNGramProfile(CommonNGramProfileTests, unittest.TestCase):

    profileClass = ngram_profile.CharNGramProfile

    def test_tokenize(self):
        self.fail('not yet implemented')


if __name__ == '__main__':
    unittest.main()
