from setuptools import setup

setup(
      install_requires=['scikit-learn',
                        'matplotlib',
                        'six',
                        'tqdm',
                        'pandas',
                        'plyfile',
                        'requests',
                        'symspellpy',
                        'termcolor',
                        'tensorboardX',
                        'shapely',
                        'pyyaml'
                        ],
      packages=['models'],
      zip_safe=False)
