# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

requirements = {
    'install': [
        'numpy',
        'logzero',
        'tqdm',
        'torch==0.4.0',
        'torchvision'
    ],
    'test': [
        'pytest',
    ],
    'docs': [
        'sphinx',
        'sphinx_rtd_theme',
        'commonmark==0.5.4',
        'recommonmark',
        'sphinx_fontawesome'
    ]
}

install_requires = requirements['install']
tests_require = requirements['test']
extras_require = {k: v for k, v in requirements.items() if k != 'install'}

setup(
    name='showcase',
    version='0.0.1',
    description='Japanese Predicate-Argument Structure Analyzer',
    long_description=readme,
    author='Yuichiroh Matsubayashi and Shun Kiyono',
    author_email='y-matsu@ecei.tohoku.ac.jp',
    url='https://github.com/cl-tohoku/showcase',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points={
        'console_scripts': ['showcase=showcase.test:main'],
    },
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.5',
    ]
)
