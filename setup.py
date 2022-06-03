from setuptools import setup, find_packages

setup_requires = []
install_requires = [
    'numpy',
    'matplotlib',
    'astropy'
]

setup(name='exgalcosutils',
      version='0.1.dev',
      url='https://github.com/hbahk/exgalcosutils',
      author='Hyeonguk Bahk',
      author_email='hyeongukbahk@gmail.com',
      description='Utilities for astronomical data handling and data analysis',
      packages=find_packages(),
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=setup_requires,
      install_requires=install_requires
      )
