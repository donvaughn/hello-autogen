# Intro to Autogen

## Project Setup

### Install python environment (Mac). One option is:
1. [Install Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)

__or__

2. `brew install --cask anaconda`
    * Requires the use of sudo password. Some workarounds [in this thread](https://stackoverflow.com/questions/42859781/best-practices-with-anaconda-and-brew)

#### Anaconda Environment
If using anaconda/miniconda, create a new environment: 

2. `conda create -n autogen python=3.11`

#### Install python dependencies
3. `pip install -r requirements.txt`

#### Install Ollama
4. `brew install ollama`

__or__

5. [Download installer](https://ollama.com/)


#### Create OpenAI Key (optional)
6. If you want to run OpenAI models (required only for example-04), create an OpenAI API key: https://platform.openai.com/api-keys
7. Copy `.env.example` to `.env` and add your API key

## Getting Started

```shell
$ npm install
$ npm start
```

will start:

- models used in the code - Ollama must be running
- all LiteLLM openAI proxies for those models
- all Panel examples:
  - http://localhost:5007/example-03-chatbot
  - http://localhost:5008/example-04-multimodal

> :warning: **IMPORTANT**: The first run will download 2 LLM models using Ollama - this will consume about 8.8GB and may take several hours depending on your connection -- and may consume your full bandwidth.

Alternatively, you can run each model separately with Ollama & LiteLLM:

```shell
# the model running on your machine
$ ollama run mistral

# the OpenAI rest API for the model (i.e. OpenAI Proxy)
$ litellm --model ollama/mistral --port 59991 --debug
```

