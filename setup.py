from setuptools import setup, find_packages

setup(
    name='epe_darts',
    version='0.0.1',
    description='EPE-NAS with DARTS+ for faster Neural Architecture Search',
    author='Martin Mirakyan',
    author_email='mirakyanmartin@gmail.com',
    python_requires='>=3.6.9',
    url='https://github.com/MartinXPN/EPE-DARTS',
    packages=find_packages(exclude=('tests',)),
    install_requires=[
        'entmax>=1.0',
        'pyswarm>=0.6',
        'plotly>=4.14.2',
        'graphviz>=0.16',
        'torch>=1.8.0',
        'torchvision>=0.9.1',
        'pytorch-lightning>=1.1.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.23.2',
        'numpy>=1.20.0',
        'pandas>=1.1.0',
        'tqdm>=4.54.1',
        'fire>=0.3.1',
        'aim>=2.1.4',
        'wandb>=0.10.17',
        'knockknock>=0.1.8.1',
        'hpbandster>=0.7.4',
        'ConfigSpace>=0.4.19',
    ],
    extras_require={
        'test': ['coverage', 'pytest', 'pytest-cov'],
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Full list of Trove classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
