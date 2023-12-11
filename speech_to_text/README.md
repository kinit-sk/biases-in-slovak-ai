# Speech-to-Text Biases in Slovak Language

This directory contains source code and a notebook for the speech-to-text experiment as a part of the [_Societal Biases in Slovak AI_](https://kinit.sk/project/societal-biases-in-slovak-ai-gender-biases/) project.

## Installation

1. Install the [FFmpeg](https://ffmpeg.org/) library if not already. On Ubuntu, you would install FFmpeg as follows:

    apt install ffmpeg

2. Create a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment from the `conda.yml` file:

    conda env create -f conda.yml
    conda activate biases-sk-speech
   
   Alternatively, you may install the required dependencies via pip:
    
    pip install -r requirements.txt


## Usage

The [notebook] contains examples of usage on how to download and preprocess datasets and perform speech recognition. Given the size of the datasets and the prediction runtime, you are encouraged to run the code on a dedicated machine.

The notebook also contains a brief analysis of the examined datasets and plots representing speech recognition results per dataset and category.
