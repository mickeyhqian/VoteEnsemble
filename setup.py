from setuptools import setup

setup(
    name='VoteEnsemble',
    version='0.1',
    py_modules=['VoteEnsemble'],
    install_requires=[
        'numpy',
        'zstandard',
    ],
    author='Huajie Qian',
    description='An implementation of the MoVE and ROVE ensembling method',
    url='https://github.com/mickeyhqian/VoteEnsemble'
)