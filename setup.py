import setuptools

DESCRIPTION = "-"
REQUIREMENTS = [i for i in open("requirements.txt").readlines()]
setuptools.setup(
    name="upjab",
    version="0.1.1",
    author="GG",
    author_email="geemguang@gmail.com",
    description=DESCRIPTION,
    # long_description = open('README.rst', encoding='utf-8').read(),
    # long_description_content_type='text/x-rst',
    python_requires='>=3.8',
    packages=['upjab'],
    # classifiers = [
    #     "Programming Language :: Python :: 3",
    #     "License :: Freely Distributable",
    #     "Operating System :: Microsoft :: Windows"
    # ],
    install_requires=REQUIREMENTS,
    # package_data={'fishmodel': ['weights/*.*']}
)