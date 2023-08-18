# Try the LLMs

You can either sign up for OpenAI developer account

Or use llama to run on local (requires beefy machines)

In this repo I mostly use the [7B](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) model.

There are also [13B](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML] and [70B](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML)

The 16GB M1 MBP can only fit 7B. 32GB can fit 13B (or llama 1 - 30B) and 64GB can fit 70B.

In terms of model quality, only 70B is close to GPT-3.5, but 13B is a good midground. 7B is quite bad.

# Instructions

1. Install python 3.10+
2. Install libraries by running `pip install -r requirements.txt`
3. Open the notebook in VS Code (or Jupyter if you're familiar). Install anything else it asks for.