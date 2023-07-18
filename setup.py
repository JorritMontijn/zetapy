from setuptools import setup

setup(
    name='zetapy',
    version='2.7.2',
    description='Calculates neuronal responsiveness index ZETA.',
    url='https://github.com/JorritMontijn/zetapy',
    author='Jorrit Montijn, Guido Meijer & Alexander Heimel',
    author_email='guido.meijer@donders.ru.nl',
    license='GNU General Public License',
    packages=['zetapy'],
    install_requires=['numpy', 'matplotlib', 'scipy'],

    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
