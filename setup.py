from setuptools import find_packages
from setuptools import setup


setup(
    name='profun',
    description='Library for protein function prediction',
    license='MIT',
    version='0.1',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'dataclasses_json', 'scikit-learn',
                      'iterative-stratification', 'scikit-optimize'],
      author='Raman Samusevich',
      author_email='raman.samusevich@gmail.com'
    )
