import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="atgm1113_keras_generic_seq2seq",
    version="1.0.0",
    description="Simple keras seq2seq enc-dec model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/attm/keras-generic-seq2seq",
    author="atgm1113",
    author_email="atgm1113@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    py_modules=['keras_generic_seq2seq'],
    include_package_data=True,
    install_requires=["tensorflow", "numpy"]
)