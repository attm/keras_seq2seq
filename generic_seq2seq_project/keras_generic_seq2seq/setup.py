import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

     name='atgm1113_generic_keras_seq2seq',  

     version='0.1',

     scripts=['keras_generic_seq2seq'] ,

     author="atgm1113",

     author_email="atgm1113@gmail.com",

     description="Simple seq2seq model made with keras",

     long_description=long_description,

   long_description_content_type="text/markdown",

     url="https://github.com/attm/keras-generic-seq2seq",

     packages=setuptools.find_packages(),

     classifiers=[

         "Programming Language :: Python :: 3",

         "License :: OSI Approved :: MIT License",

         "Operating System :: OS Independent",

     ],

 )