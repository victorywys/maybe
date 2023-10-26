from setuptools import setup, find_packages

setup(
    name='maybee',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your package's dependencies here, e.g.,
        # 'numpy>=1.10',
        'torch>=2.0',
        'utilsd',
        'pandas',
        'scipy',
        'tqdm',
        'joblib',
    ],
    # other metadata like author, license, etc.
    author='RiichiUra3',
    author_email='your.email@example.com',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
