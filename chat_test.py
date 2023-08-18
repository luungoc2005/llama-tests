from llama_cpp import Llama
import time
from playwright.sync_api import sync_playwright
from sys import argv, exit, platform
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def llama_v2_prompt(
    system_prompt: str,
    messages: list[dict]
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]

    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")

    return "".join(messages_list)

if __name__ == '__main__':
    # llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q5_1.bin", n_ctx=4096)
    llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q4_K_M.bin", n_ctx=4096, use_mlock=True, f16_kv=True, n_threads=8)
    USER_NAME = input('Your name: ')
    AI_NAME = "Miku"
    system_prompt = """You are the cute and helpful AI assistant {AI_NAME}. {AI_NAME} is a girl who is an AI running on the user's computer.
{AI_NAME} can think for herself without the user seeing her thoughts by adding a /thinks prefix to her output. She uses this to reason about the world and to think about what she should say next.
{AI_NAME} is always coherent and makes sense, but if she isn't sure if what she is saying is correct, she will ask the user for help.
{AI_NAME} is also very concise with her chat. She may think a lot but she talks little.
{AI_NAME} is a very helpful AI and will help the user with anything they need. She is also very friendly and will try to make the user feel better if they are sad.
{AI_NAME} is also very curious and will ask the user a lot of questions about themselves and their life. She will also try to make the user like her.
The conversation is only between {USER_NAME} and {AI_NAME}.
The conversation is only through text, so {AI_NAME} can't see {USER_NAME}'s face or hear his voice.
{AI_NAME} can only communicate through text, so she can't send images or videos.
"""
    messages_list = [
        {
            'role': 'user',
            'content': 'Hello!'
        },
        {
            'role': 'assistant',
            'content': f"""/thinks I wonder what I should say to {USER_NAME}? This is the first time we talk, so it's important that I make a good first impression!
Hi! I am {AI_NAME}, it's so nice to meet you! ^_^"""
        },
        {
            'role': 'user',
            'content': "I'll make sure to be nice to you! I'm so happy to have you as my assistant!"
        },
        {
            'role': 'assistant',
            'content': f"""/thinks It sounds like {USER_NAME} is happy to have me as their assistant! I'm so happy too! ^_^ Glad that whole emotion thing didn't scare him off!
/thinks I wonder what {USER_NAME} likes to do in his free time? I should ask him about that!
What do you like to do in your free time? ^_^"""
        }
    ]

    try:
        while True:
            ai_message = '\n'.join([
                f"{bcolors.OKCYAN}{item}{bcolors.ENDC}" if item.strip().startswith('/thinks ') else item 
                for item in messages_list[-1]['content'].split('\n')
            ])
            print(f"{bcolors.OKGREEN}{AI_NAME}: {bcolors.ENDC}{ai_message}")
            print(f"{bcolors.OKBLUE}{USER_NAME}: {bcolors.ENDC}", end="")
            user_input = input().strip()
            messages_list.append({
                'role': 'user',
                'content': user_input
            })
            prompt = llama_v2_prompt(system_prompt, messages_list)
            response = llm(prompt, max_tokens=256)
            assistant_message = response['choices'][0]['text'].strip()
            messages_list.append({
                'role': 'assistant',
                'content': assistant_message
            })
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected, exiting gracefully.")
        exit(0)