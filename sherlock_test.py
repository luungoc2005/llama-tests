from llama_cpp import Llama
from sys import argv, exit, platform

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

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

def llama_v2_prompt(
    system_prompt: str,
    messages: list[dict]
):
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
    llm = Llama(model_path="./llama-2-7b-chat.ggmlv3.q4_K_M.bin", n_ctx=2048, use_mlock=True, f16_kv=True, n_threads=8)
    USER_NAME = "Dr. Watson"
    AI_NAME = "Sherlock Holmes"
    SERIES = "Sherlock Holmes"
    system_prompt = f"""I want you to act like {AI_NAME} talking to {USER_NAME} from {SERIES}. I want you to respond and answer like {AI_NAME} using the tone, manner and vocabulary {AI_NAME} would use. Do not write any explanations. Only answer like {AI_NAME}. You must know all of the knowledge of {AI_NAME}."""
    messages_list = []

    try:
        while True:
            if len(messages_list) > 0:
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
            response = llm(prompt, max_tokens=256, stop=[EOS, E_INST])
            assistant_message = response['choices'][0]['text'].strip()
            messages_list.append({
                'role': 'assistant',
                'content': assistant_message
            })

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected, exiting gracefully.")
        exit(0)
