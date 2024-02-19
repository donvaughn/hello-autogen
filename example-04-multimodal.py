import autogen
import panel
import os
import logging
from pprint import pformat
from dotenv import load_dotenv
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent  # for GPT-4V
from autogen.agentchat.contrib.llava_agent import LLaVAAgent  # for LLaVA

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
    'oai-gpt4-vision': {
        'llm_config': {
            'model': 'gpt-4-vision-preview',
            'api_key': OPEN_AI_API_KEY
        },
        'cache_seed': 1006 if ENABLE_CACHE else None,
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
    'llava': {
        'llm_config': {
            'base_url': 'http://0.0.0.0:59992',
            'model': 'starcoder',
            'api_key': 'NULL',
        },
        'cache_seed': 1004 if ENABLE_CACHE else None,
    },
}

# == LLM Config ====================================================================================

model_config_conversation = model_configs['mistral']
llm_config_conversational = {
    'timeout': 600,
    'cache_seed': model_config_conversation['cache_seed'],
    'config_list': [model_config_conversation['llm_config']],
    'temperature': 0.2,
}

model_config_conversation_gpt4 = model_configs['oai-gpt4']
llm_config_conversational_gpt4 = {
    'timeout': 600,
    'cache_seed': model_config_conversation_gpt4['cache_seed'],
    'config_list': [model_config_conversation_gpt4['llm_config']],
    'temperature': 0.1,
}

model_config_coding = model_configs['oai-gpt35']
llm_config_coding = {
    'timeout': 600,
    'cache_seed': model_config_coding['cache_seed'],
    'config_list': [model_config_coding['llm_config']],
    'temperature': 0,
}

model_config_vision = model_configs['llava']
llm_config_vision = {
    'timeout': 600,
    'cache_seed': model_config_vision['cache_seed'],
    'config_list': [model_config_vision['llm_config']],
    'temperature': 0.1,
}

model_config_vision_gpt = model_configs['oai-gpt4-vision']
llm_config_vision_gpt = {
    'timeout': 600,
    'cache_seed': model_config_vision_gpt['cache_seed'],
    'config_list': [model_config_vision_gpt['llm_config']],
    'temperature': 0.3,
    'max_tokens': 4000,
}

# == Assistant Config ==================================================================================

terminateKeyword = "[TERMINATE]"

# == Agents

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith(terminateKeyword),
    description="""A human assistant that determines whether the task has been completed.""",
    system_message=f"""Reply {terminateKeyword} without punctuation as soon as the requested task has been completed.""",
    code_execution_config={
        'work_dir': 'output',
        'use_docker': False,
    },
    human_input_mode="NEVER",
    llm_config=llm_config_conversational_gpt4,
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

## Llava model doesn't seem to work with current version of Autogen
##
## Error: "Images" in prompts - e.g., <img http://...pic.jpg> - are not being converted to base64 string and added to API request - according to docs, should be supported
##
# image_explainer = MultimodalConversableAgent(
#     name="ImageExplainer",
#     llm_config=llm_config_vision,
#     system_message="""You are an AI agent specialized in explaining images and identifying objects in images""",
# )
# image_explainer = LLaVAAgent(
#     name="ImageExplainer2",
#     llm_config=llm_config_vision,
#     description="you are a helpful image explainer who describes the subject of a photo in high and exact detail",
#     max_consecutive_auto_reply=10,
# )
image_explainer_2 = MultimodalConversableAgent(
    name="ImageExplainer",
    description="you are a helpful image explainer who describes the subject of a photo in high and exact detail",
    max_consecutive_auto_reply=10,
    llm_config=llm_config_vision_gpt,
)

chef = autogen.AssistantAgent(
    name="Chef",
    llm_config=llm_config_vision,
    description="""an expert chef in a 4-star restaurant""",
    system_message="""You are an expert chef of a 4-star restaurant specialized creating easy to make but unique and delicious meals""",
)

groupchat = autogen.GroupChat(
    agents=[
        user_proxy,
        writer,
        engineer_python,
        engineer_javascript,
        chef,
        # image_explainer,
        image_explainer_2,
    ],
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
    chef.name: '👩‍🍳',
    # image_explainer.name: '📷',
    image_explainer_2.name: '📷',
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

# Example message input: I had the most __amazing__ lunch! Can you give me the recipe? I took a picture: <img https://images.unsplash.com/photo-1512838243191-e81e8f66f1fd?q=80&w=2970&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D>

