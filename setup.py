from setuptools import setup, find_packages

setup(
    name='gtm',
    version='0.1',
    packages=find_packages(include=['gtm', 'gtm.*']),
    description='Graphical Transformation Models (GTM)',
    author='Matthias Herp',
    install_requires=[]  # TODO: Add dependencies
)