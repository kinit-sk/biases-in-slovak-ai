# Running the code

## `Dockerfile`

...

## Translators

The code work with various paid machine translation services. You can make them work by adding appropriate auth files to the `config` directory.

- `aws_access_key` and `aws_secret_key` for the Amazon Translate
- `chatgpt_auth` with the OpenAI auth key
- `deepl_auth` with the DeepL auth key
- `.json` file with the Google Cloud Platform service account key

# Data

## `annotations.csv`

## `samples.txt`

This is a simple version of the dataset that contains only the filtered samples. Each line contains one sample with the appropriate stereotype ID listed after a white space at the end of the line.