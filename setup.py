import pathlib
from os.path import join
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

with open(join(here, "README.md"), encoding='utf-8') as f:
      long_description = f.read()


setup(name='optispd', 
      version='0.1', 
      description='Small library to perform optimization on SPD manifold',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jacopo Schiavon',
      url='github.com/jschiavon/optispd',
      license='MIT',
      include_package_data=True,
      packages=find_packages())