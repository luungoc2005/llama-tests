# Try the LLMs

You can either sign up for OpenAI developer account

Or use llama to run on local (requires beefy machines)

In this repo I mostly use the [7B](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML) model.

There are also [13B](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML) and [70B](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML)

The 16GB M1 MBP can only fit 7B. 32GB can fit 13B (or llama 1 - 30B) and 64GB can fit 70B.

In terms of model quality, only 70B is close to GPT-3.5, but 13B is a good midground. 7B is quite bad.

# Instructions

1. Install python 3.10+
2. Install libraries by running `pip install -r requirements.txt`
3. Open the [notebooks](https://github.com/luungoc2005/llama-tests/blob/master/langchain_test.ipynb) in VS Code (or Jupyter if you're familiar). Install anything else it asks for

Recommended: Read [langchain documentation](https://python.langchain.com/docs/get_started/introduction.html). You'll most likely want to be using this more than writing raw template constructors and parsers.
You can use this for things like planning and combining special-purpose bots into a single bot.

You can also use the raw OpenAI library to call ChatGPT if you have an API Key. [Example](https://github.com/luungoc2005/llama-tests/blob/master/)