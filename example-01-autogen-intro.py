import autogen
import logging
from pprint import pformat
import os
from dotenv import load_dotenv

load_dotenv()
OPEN_AI_API_KEY = os.getenv('OPEN_AI_API_KEY')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# == Model Config ====================================================================================

model_configs = {
    'mistral': {
        'llm_config': {
            'base_url': 'http://0.0.0.0:59991',
            'model': 'mistral',
            'api_key': 'NULL',
        },
        'cache_seed': 1000,
        # 'cache_seed': False, # disables cache
    },
    'oai-gpt35': {
        'llm_config': {
            'model': 'gpt-3.5-turbo-16k',
            'api_key': OPEN_AI_API_KEY
        },
        'cache_seed':  1001,
    },
    'oai-gpt4': {
        'llm_config': {
            'model': 'gpt-4-turbo-preview',
            'api_key': OPEN_AI_API_KEY
        },
        'cache_seed': 1002,
    },
}

# == LLM Config ====================================================================================

model_config_conversation = model_configs['mistral']
llm_config_conversational = {
    'timeout': 600,
    'cache_seed': model_config_conversation['cache_seed'],
    'config_list': [model_config_conversation['llm_config']],
    'temperature': 0.25,
}

model_config_conversation_gpt35 = model_configs['oai-gpt35']
llm_config_conversational_gpt35 = {
    'timeout': 600,
    'cache_seed': model_config_conversation_gpt35['cache_seed'],
    'config_list': [model_config_conversation_gpt35['llm_config']],
    'temperature': 0.25,
}

model_config_conversation_gpt4 = model_configs['oai-gpt4']
llm_config_conversational_gpt4 = {
    'timeout': 600,
    'cache_seed': model_config_conversation_gpt4['cache_seed'],
    'config_list': [model_config_conversation_gpt4['llm_config']],
    'temperature': 0.25,
}

# == Assistant Config ==================================================================================

terminateKeyword = "TERMINATE"

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(terminateKeyword),
    description="""an assistant with strong communication skills""",
    system_message=f"""Reply {terminateKeyword} if the task has been solved at full satisfaction""",
    code_execution_config={
        'work_dir': 'output',
        'use_docker': False,
    },
    human_input_mode="NEVER",
    llm_config=llm_config_conversational,
    # llm_config=llm_config_conversational_gpt35,
    max_consecutive_auto_reply=10,
)

assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config=llm_config_conversational,
    # llm_config=llm_config_conversational_gpt35,
    # llm_config=llm_config_conversational_gpt4,
    description="""an helpful assistant with strong writing skills who can communicate clearly and without fluff""",
    system_message="""You are a senior editor and acclaimed writer and researcher""",
)

# == Prompt ====================================================================================

with_termination_notice = lambda task: task + (
    ''
    '\n\nDo not show appreciation in your responses, say only what is necessary. '
    'if "Thank you" or "You\'re welcome" are said in the conversation, then say ' + terminateKeyword + ' '
    'to indicate the conversation is finished and this is your last message.'
)

task = with_termination_notice("""Tell me a joke.""")
# task = with_termination_notice("""Write python code to output numbers from 1 to 100 then store the python code in a file.""")

# == Chat Execution ====================================================================================

result = user_proxy.initiate_chat(
    assistant,
    message=task
)

logging.info(pformat(result))
