import os
import sys
sys.path.insert(0, './Resources')
import time
from datetime import datetime
from uuid import uuid4
import json
import importlib.util
import requests
import tkinter as tk
import asyncio
import aiofiles
import re


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
       return file.read().strip()
    
    
def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.datetime.fromtimestamp(unix_time)
    datetime_str = datetime_obj.strftime("%A, %B %d, %Y at %I:%M%p %Z")
    return datetime_str

def import_api_function():
    settings_path = './Settings.json'
    with open(settings_path, 'r') as file:
        settings = json.load(file)
    api_module_name = settings['API']
    module_path = f'./Resources/API_Calls/{api_module_name}.py'
    spec = importlib.util.spec_from_file_location(api_module_name, module_path)
    api_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_module)
    llm_api_call = getattr(api_module, 'LLM_API_Call', None)
    agent_llm_api_call = getattr(api_module, 'Agent_LLM_API_Call', None)
    input_expansion_api_call = getattr(api_module, 'Input_Expansion_API_Call', None)
    domain_selection_api_call = getattr(api_module, 'Domain_Selection_API_Call', None)
    db_prune_api_call = getattr(api_module, 'DB_Prune_API_Call', None)
    inner_monologue_api_call = getattr(api_module, 'Inner_Monologue_API_Call', None)
    intuition_api_call = getattr(api_module, 'Intuition_API_Call', None)
    final_response_api_call = getattr(api_module, 'Final_Response_API_Call', None)
    short_term_memory_response_api_call = getattr(api_module, 'Short_Term_Memory_API_Call', None)
    if llm_api_call is None:
        raise ImportError(f"LLM_API_Call function not found in {api_module_name}.py")
    return llm_api_call, agent_llm_api_call, input_expansion_api_call, domain_selection_api_call, db_prune_api_call, inner_monologue_api_call, intuition_api_call, final_response_api_call, short_term_memory_response_api_call

# Create a Description for the Module
def Database_Search_Description(username, bot_name):
    description = f"Database_Search.py: This tool is designed to search a database of uploaded documentation to provide accurate and relevant information for your inquiries. It meticulously scans the stored documents, ensuring that the extracted information is precise and applicable to your specific needs."
    return description
    
    
# Add your custom code here, name must be the same as file name.
async def Database_Search(host, bot_name, username, user_id, task, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input):
    try:
        async with aiofiles.open('./Settings.json', mode='r', encoding='utf-8') as f:
            settings = json.loads(await f.read())
    #    embed_size = settings['embed_size']
        Sub_Module_Output = settings.get('Output_Sub_Module', 'False')
        vector_db = settings.get('Vector_DB', 'Qdrant_DB')
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        API = settings.get('API', 'Oobabooga')
        tasklist_completion2 = list()
        conversation = list()
        memcheck  = list()
        memcheck2 = list()
        sub_agent_completion = list()
        botnameupper = bot_name.upper()
        usernameupper = username.upper()
        task_completion = "Task Failed"
        LLM_API_Call, Agent_LLM_API_Call, Input_Expansion_API_Call, Domain_Selection_API_Call, DB_Prune_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
        try:
            collection_name = f"BOT_NAME_{bot_name}"
            db_search_module_name = f'Resources.DB_Search.{vector_db}'
            db_search_module = importlib.import_module(db_search_module_name)
            client = db_search_module.initialize_client()
            domain = "test"
            result = db_search_module.search_db(collection_name, bot_name, user_id, task, domain, search_number=30)
            
            conversation.append({'role': 'assistant', 'content': f"MEMORIES: {result}\n\n"})

        except Exception as e:
            print(e)
            
        task = re.sub(r'\[.*?\]:\s*', '', task)

            
        conversation.append({'role': 'user', 'content': f"SYSTEM: Summarize the pertinent information from the given memories related to the given task. Present the summarized data in a single, easy-to-understand paragraph. Do not generalize, expand upon, or use any latent knowledge in your summary, only return a truncated version of previously given information."})
        conversation.append({'role': 'assistant', 'content': f"BOT {task_counter}: Sure, here's an overview of the scraped text: "})

        if API == "OpenAi":
            task_completion = await Agent_LLM_API_Call(API, backend_model, conversation, username, bot_name)
        if API == "Oobabooga":
            task_completion = await Agent_LLM_API_Call(host, API, backend_model, conversation, username, bot_name)
        if API == "KoboldCpp":
            task_completion = await Agent_LLM_API_Call(host, API, backend_model, conversation, username, bot_name)
        if API == "AetherNode":
            prompt = ''.join([message_dict['content'] for message_dict in conversation])
            task_completion = await Agent_LLM_API_Call(host, API, prompt, username, bot_name)
        
        conversation.clear()
        sub_agent_completion.append({'role': 'user', 'content': f"TASK {task_counter}: {task}"})
        sub_agent_completion.append({'role': 'assistant', 'content': f"COMPLETED TASK {task_counter}: {task_completion}"})
        return sub_agent_completion


    except Exception as e:
        print(f'Failed with error: {e}')
        error = 'ERROR WITH PROCESS LINE FUNCTION'
        return error
