import autogen
import panel
import os
import logging
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

model_config_conversation_gpt4 = model_configs['oai-gpt4']
llm_config_conversational_gpt4 = {
    'timeout': 600,
    'cache_seed': model_config_conversation_gpt4['cache_seed'],
    'config_list': [model_config_conversation_gpt4['llm_config']],
    'temperature': 0,
}

model_config_coding = model_configs['oai-gpt35']
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
    description="""A project manager with strong communication skills that only interacts when assistants cannot answer or to terminate chat""",
    system_message=f"""Reply {terminateKeyword} without punctuation if the task has been solved at full satisfaction. You only interact when assistants cannot answer satisfactorily or to terminate conversation""",
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
    description="""a helpful assistant with strong writing skills who can communicate clearly and without fluff""",
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
    llm_config=llm_config_conversational_gpt4
)

# == Prompt ====================================================================================

with_termination_notice = lambda task: task + (
        '\n\nDo not show appreciation in your responses, say only what is necessary. '
        'if "Thank you" or "You\'re welcome" are said in the conversation, then say ' + terminateKeyword + ' '
                                                                                                           'to indicate the conversation is finished and this is your last message.'
)

# === Panel integration ===========================================================================
# === Thanks: https://github.com/yeyu2/Youtube_demos/blob/main/panel_autogen_2.py

avatar = {
    user_proxy.name: "👨‍💼",
    writer.name: "👩‍💻",
    engineer_python.name: "👩‍🔬",
    engineer_javascript.name: "👨‍🚀",
}

# Print each agent message to chat window
def print_messages(recipient, messages, sender, config):
    content = messages[-1]['content']

    if all(key in messages[-1] for key in ['name']):
        chat_interface.send(content, user=messages[-1]['name'], avatar=avatar[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(content, user=recipient.name, avatar=avatar[recipient.name], respond=False)

    # tells autogen to continue agent communication
    return False, None

# Kick off an autogen chat sequence on each message entered into chat UI & print cost message at end of sequence
def perform_chat_sequence(contents: str, user: str, instance: panel.chat.ChatInterface):
    result = user_proxy.initiate_chat(manager, message=with_termination_notice(contents))
    total_cost = 0

    for costInfo in result.cost:
        total_cost += costInfo['total_cost']
    total_cost_dollars = '${:,.2f}'.format(total_cost)
    chat_interface.send(total_cost_dollars, user="Accountant", avatar="🤑", respond=False)

# == Register message sending with each agent

user_proxy.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)

writer.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
engineer_python.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
engineer_javascript.register_reply(
    [autogen.Agent, None],
    reply_func=print_messages,
    config={"callback": None},
)
panel.extension(design="material")

# == Start chat UI

chat_interface = panel.chat.ChatInterface(callback=perform_chat_sequence)
chat_interface.send("Ready to assist!", user="System", respond=False)
chat_interface.servable()
