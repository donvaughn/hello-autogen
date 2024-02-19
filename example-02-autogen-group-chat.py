import autogen
import os
import logging
from pprint import pformat
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

# ENABLE_CACHE = True
ENABLE_CACHE = False

# == Model Config ====================================================================================

model_configs = {
    'oai-gpt35': {
        'llm_config': {
            'model': 'gpt-3.5-turbo-16k',
            'api_key': OPEN_AI_API_KEY
        },
        'cache_seed':  1000 if ENABLE_CACHE else None,
    },
    'oai-gpt4': {
        'llm_config': {
            'model': 'gpt-4-turbo-preview',
            'api_key': OPEN_AI_API_KEY
        },
        'cache_seed': 1001 if ENABLE_CACHE else None,
    },
    'mistral': {
        'llm_config': {
            'base_url': 'http://0.0.0.0:59991',
            'model': 'mistral',
            'api_key': 'NULL',
        },
        'cache_seed': 1003 if ENABLE_CACHE else None,
    },
    'codellama': {
        'llm_config': {
            'base_url': 'http://0.0.0.0:59993',
            'model': 'codellama',
            'api_key': 'NULL',
        },
        'cache_seed': 1005 if ENABLE_CACHE else None,
    },
}

# == LLM Config ====================================================================================

model_config_conversation = model_configs['mistral']
llm_config_conversational = {
    'timeout': 600,
    'cache_seed': model_config_conversation['cache_seed'],
    'config_list': [model_config_conversation['llm_config']],
    'temperature': 0,
}

model_config_conversation_advanced = model_configs['oai-gpt4']
llm_config_conversational_advanced = {
    'timeout': 600,
    'cache_seed': model_config_conversation_advanced['cache_seed'],
    'config_list': [model_config_conversation_advanced['llm_config']],
    'temperature': 0,
}

model_config_coding = model_configs['oai-gpt4']
llm_config_coding = {
    'timeout': 600,
    'cache_seed': model_config_coding['cache_seed'],
    'config_list': [model_config_coding['llm_config']],
    'temperature': 0,
}

# == Assistant Config ==================================================================================

terminateKeyword = "[TERMINATE]"

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(terminateKeyword),
    description="""an assistant with strong communication skills""",
    system_message=f"""Reply {terminateKeyword} without punctuation if the task has been solved at full satisfaction""",
    code_execution_config={
        'work_dir': 'output',
        'use_docker': False,
    },
    human_input_mode="NEVER",
    llm_config=llm_config_conversational,
)

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config_conversational,
    description="""an helpful assistant with strong writing skills who can communicate clearly and without fluff""",
    system_message="""You are a senior editor and acclaimed writer with exceptional skill in engaging and concise storytelling""",
)

engineer_python = autogen.AssistantAgent(
    name="PythonEngineer",
    llm_config=llm_config_coding,
    description="""an assistant with strong software engineering skills specialized in python programming language""",
    system_message="""You are a senior python engineer.""",
)

engineer_javascript = autogen.AssistantAgent(
    name="JavascriptEngineer",
    llm_config=llm_config_coding,
    description="""an assistant with strong software engineering skills specialized in javascript programming language""",
    system_message="""You are a senior javascript engineer.""",
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, writer, engineer_python, engineer_javascript],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_conversational_advanced
)

# == Prompt ====================================================================================

with_termination_notice = lambda task: task + (
    '\n\nDo not show appreciation in your responses, say only what is necessary. '
    'if "Thank you" or "You\'re welcome" are said in the conversation, then say ' + terminateKeyword + ' '
    'to indicate the conversation is finished and this is your last message.'
)

task = with_termination_notice("""Tell me a very short story.""")
# task = with_termination_notice("""Write python code to output numbers from 1 to 100""")
# task = with_termination_notice("""Write python code to output numbers from 1 to 100 then store the python code in a file.""")
# task = with_termination_notice("""Write javascript function to output numbers from 1 to n""")
# task = with_termination_notice("""Write a react js function to show 1 to n boxes on the screen. the variable n should come from a prop call "count" """)
# task = with_termination_notice("""How do I create a dataframe? """)

# == Chat Execution ====================================================================================

result = user_proxy.initiate_chat(
    manager,
    message=task
)

logging.info(pformat(result))
