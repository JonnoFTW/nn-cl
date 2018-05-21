from distutils.core import setup

setup(
    name='nn-cl',
    version='0.1dev',
    packages=['nncl', ],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    requires=open('requirements.txt').readlines()
)
