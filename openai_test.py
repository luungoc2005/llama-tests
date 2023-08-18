from tqdm import tqdm
import tiktoken
import pyperclip
import argparse
import openai
import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL_ID = 'gpt-3.5-turbo'
PROMPT_MAX_LEN = 2000

openai.api_key = OPENAI_API_KEY


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def translate(text, from_language):
    def construct_messages(single_text):
        return [
            {"role": "system", "content": f"You are a helpful assistant that translates {from_language} to English"},
            {"role": "user", "content": f"Translate the following {from_language} text to English: {single_text}"}
        ]

    messages = construct_messages(text)
    # 4096 tokens max including prompt, so if prompt is more than 1500 we break the message down
    if num_tokens_from_messages(messages) < PROMPT_MAX_LEN:
        response = openai.ChatCompletion.create(
            model=MODEL_ID,
            messages=messages
        )
        return response.choices[0].message.content
    else:
        paragraphs = text.split("\n")
        max_len = -1
        splits = 1
        messages_list = []
        while max_len == -1 or max_len > PROMPT_MAX_LEN:
            splits += 1
            cutoff_length = len(paragraphs) // splits
            message_batches = [paragraphs[i:i+cutoff_length]
                               for i in range(0, len(paragraphs), cutoff_length)]
            messages_list = [construct_messages(
                "\n".join(batch)) for batch in message_batches]
            messages_len = [num_tokens_from_messages(
                item) for item in messages_list]
            max_len = max(messages_len)

        final_response = ''
        for item in tqdm(messages_list):
            response = openai.ChatCompletion.create(
                model=MODEL_ID,
                messages=item
            )
            final_response += response.choices[0].message.content + '\n'

        return final_response

# models = [item.id for item in openai.Model.list().data]
# print(models)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', required=False)
    parser.add_argument('--from_language', type=str,
                        default='Japanese', required=False)
    args = parser.parse_args()

    if args.input == '':
        text = pyperclip.paste()
    else:
        with open(args.input, 'r') as f:
            text = f.read()

    response = translate(text, args.from_language)

    print(response)
