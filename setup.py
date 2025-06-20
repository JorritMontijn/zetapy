from setuptools import setup, find_packages

setup(
    name='zetapy',
    version='3.2.1',
    description='Implementations of the ZETA family of statistical tests.',
    url='https://github.com/JorritMontijn/zetapy',
    author='Jorrit Montijn, Guido Meijer & Alexander Heimel',
    author_email='j.s.montijn@gmail.com',
    license='GPL-3.0-only',
    packages=find_packages(),
    install_requires=[
        'scipy >= 1.1.0',
        'numpy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    include_package_data=True,
)