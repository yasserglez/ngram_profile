from distutils.core import setup

import ngram_profile


setup(name='ngram_profile',
      version=ngram_profile.__version__,
      url='https://github.com/yasserglez/ngram_profile',
      description='Text classification based on character n-grams.',
      author='Yasser Gonzalez',
      author_email='contact@yassergonzalez.com',
      classifiers=[
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Development Status :: 5 - Production/Stable',
          'License :: OSI Approved :: Apache Software License',
      ],
      py_modules=['ngram_profile'],
      install_requires=[
          'six >= 1.9.0',
      ])
