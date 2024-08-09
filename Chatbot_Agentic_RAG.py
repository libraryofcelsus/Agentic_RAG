import sys
import os
import json
import time
import datetime as dt
from datetime import datetime
from uuid import uuid4
import requests
import shutil
import importlib
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import re
import keyboard
import traceback
import asyncio
import aiofiles
import aiohttp
import base64


Debug_Output = "True"
Memory_Output = "False"
Dataset_Output = "False"

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
       return file.read().strip()

def timestamp_func():
    try:
        return time.time()
    except:
        return time()

def is_url(string):
    return string.startswith('http://') or string.startswith('https://')

def timestamp_to_datetime(unix_time):
    datetime_obj = datetime.fromtimestamp(unix_time)
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

def load_format_settings(backend_model):
    file_path = f'./Model_Formats/{backend_model}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            formats = json.load(file)
    else:
        formats = {
            "heuristic_input_start": "",
            "heuristic_input_end": "",
            "system_input_start": "",
            "system_input_end": "",
            "user_input_start": "", 
            "user_input_end": "", 
            "assistant_input_start": "", 
            "assistant_input_end": ""
        }
    return formats

def set_format_variables(backend_model):
    format_settings = load_format_settings(backend_model)
    heuristic_input_start = format_settings.get("heuristic_input_start", "")
    heuristic_input_end = format_settings.get("heuristic_input_end", "")
    system_input_start = format_settings.get("system_input_start", "")
    system_input_end = format_settings.get("system_input_end", "")
    user_input_start = format_settings.get("user_input_start", "")
    user_input_end = format_settings.get("user_input_end", "")
    assistant_input_start = format_settings.get("assistant_input_start", "")
    assistant_input_end = format_settings.get("assistant_input_end", "")

    return heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end
    
def format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, response):
    try:
        if response is None:
            return "ERROR WITH API"  
        if backend_model == "Llama_3":
            assistant_input_start = "assistant"
            assistant_input_end = "assistant"
        botname_check = f"{botnameupper}:"
        while (response.startswith(assistant_input_start) or response.startswith('\n') or
               response.startswith(' ') or response.startswith(botname_check)):
            if response.startswith(assistant_input_start):
                response = response[len(assistant_input_start):]
            elif response.startswith(botname_check):
                response = response[len(botname_check):]
            elif response.startswith('\n'):
                response = response[1:]
            elif response.startswith(' '):
                response = response[1:]
            response = response.strip()
        botname_check = f"{botnameupper}: "
        if response.startswith(botname_check):
            response = response[len(botname_check):].strip()
        if backend_model == "Llama_3":
            if "assistant\n" in response:
                index = response.find("assistant\n")
                response = response[:index]
        if response.endswith(assistant_input_end):
            response = response[:-len(assistant_input_end)].strip()
        
        return response
    except:
        traceback.print_exc()
        return ""  
        
    
    
async def load_filenames_and_descriptions(folder_path, username, user_id, bot_name):
    """
    Load all Python filenames in the given folder along with their descriptions.
    Returns a dictionary mapping filenames to their descriptions.
    """
    filename_description_map = {}
    
    def extract_function_code(code, func_name):
        """
        Extract the function definition from the code.
        """
        lines = code.splitlines()
        func_lines = []
        inside_func = False
        indent_level = None
        
        for line in lines:
            if inside_func:
                if line.startswith(" " * indent_level) or line.strip() == "":
                    func_lines.append(line)
                else:
                    break
            elif line.strip().startswith(f"def {func_name}("):
                inside_func = True
                indent_level = len(line) - len(line.lstrip())
                func_lines.append(line)
        
        return "\n".join(func_lines) if func_lines else None

    try:
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.py')]
        
        for filename in filenames:
            base_filename = os.path.splitext(filename)[0]
            module_path = os.path.join(folder_path, filename)
            
            with open(module_path, 'r') as file:
                code = file.read()
            
            description_function_name = f"{base_filename}_Description"
            description_function_code = extract_function_code(code, description_function_name)
            
            description = "Description function not found."
            if description_function_code:
                try:
                    local_scope = {}
                    exec(description_function_code, globals(), local_scope)
                    description = local_scope[description_function_name](username, bot_name)
                    description = description.replace('<<USERNAME>>', username)
                    description = description.replace('<<BOTNAME>>', bot_name)
                except Exception as e:
                    print(f"An error occurred: {e}")
            filename_description_map[filename] = {"filename": filename, "description": description}
                
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return filename_description_map
    
    
    
async def load_folder_names_and_descriptions(folder_path):
    """
    Load all folder names in the given folder along with their descriptions from a .txt file.
    Returns a dictionary mapping folder names to their descriptions.
    """
    folder_description_map = {}
    try:
        folders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for folder in folders:
            description_file_path = os.path.join(folder_path, folder, f"{folder}.txt")
            if os.path.isfile(description_file_path):
                with open(description_file_path, 'r') as file:
                    description = file.read().strip()
            else:
                description = "Description file not found."
            folder_description_map[folder] = description
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return folder_description_map

    
def write_dataset_simple(backend_model, user_input, output):
    data = {
        "input": user_input,
        "output": output
    }
    try:
        with open(f'{backend_model}_simple_dataset.json', 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(f'{backend_model}_simple_dataset.json', 'w') as file:
            json.dump([data], file, indent=4)


class MainConversation:
    def __init__(self, username, user_id, bot_name, max_entries):
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        self.format_config = self.initialize_format(backend_model)
        
        self.bot_name_upper = bot_name.upper()
        self.username_upper = username.upper()
        self.max_entries = int(max_entries)
        self.file_path = f'./History/{user_id}/{bot_name}_Conversation_History.json'
        self.main_conversation = [] 
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.running_conversation = data.get('running_conversation', [])
        else:
            self.running_conversation = []
            self.save_to_file()

    def initialize_format(self, backend_model):
        file_path = f'./Model_Formats/{backend_model}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                formats = json.load(file)
        else:
            formats = {
                "user_input_start": "", 
                "user_input_end": "", 
                "assistant_input_start": "", 
                "assistant_input_end": ""
            }
        return formats

    def format_entry(self, user_input, response, initial=False):
        user = f"{self.username_upper}: {user_input}"
        bot = f"{self.bot_name_upper}: {response}"
        return {'user': user, 'bot': bot}

    def append(self, timestring, user_input, response):
        entry = self.format_entry(f"[{timestring}] - {user_input}", response)
        self.running_conversation.append(entry)
        while len(self.running_conversation) > self.max_entries:
            self.running_conversation.pop(0)
        self.save_to_file()

    def save_to_file(self):
        data_to_save = {
            'main_conversation': self.main_conversation,
            'running_conversation': self.running_conversation
        }
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def get_conversation_history(self):
        formatted_history = []
        for entry in self.running_conversation: 
            user_entry = entry['user']
            bot_entry = entry['bot']
            formatted_history.append(user_entry)
            formatted_history.append(bot_entry)
        return '\n'.join(formatted_history)
        
    def get_dict_conversation_history(self):
        formatted_history = []
        for entry in self.running_conversation:
            user_entry = {'role': 'system', 'content': entry['user']}
            bot_entry = {'role': 'assistant', 'content': entry['bot']}
            formatted_history.append(user_entry)
            formatted_history.append(bot_entry)
        return formatted_history

    def get_dict_formated_conversation_history(self, user_input_start, user_input_end, assistant_input_start, assistant_input_end):
        formatted_history = []
        for entry in self.running_conversation:
            user_entry = {'role': 'user', 'content': f"{user_input_start}{entry['user']}{user_input_end}"}
            bot_entry = {'role': 'assistant', 'content': f"{assistant_input_start}{entry['bot']}{assistant_input_end}"}
        return formatted_history

    def get_last_entry(self):
        if self.running_conversation:
            return self.running_conversation[-1]
        return None
    
    def delete_conversation_history(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            self.running_conversation = []
            self.save_to_file()


async def Agentic_RAG_Chatbot(user_input, username, user_id, bot_name, image_path=None):
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    API = settings.get('API', 'Oobabooga')
    conv_length = settings.get('Conversation_Length', '3')
    backend_model = settings.get('Model_Backend', 'Llama_3')
    LLM_Model = settings.get('LLM_Model', 'Oobabooga')
    Web_Search = settings.get('Search_Web', 'False')
    Write_Dataset = settings.get('Write_To_Dataset', 'False')
    Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Simple')
    Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
    vector_db = settings.get('Vector_DB', 'Qdrant_DB')
    LLM_API_Call, Agent_LLM_API_Call, Input_Expansion_API_Call, Domain_Selection_API_Call, DB_Prune_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
    heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
    end_prompt = ""
    base_path = "./Chatbot_Prompts"
    base_prompts_path = os.path.join(base_path, "Base")
    user_bot_path = os.path.join(base_path, user_id, bot_name)  
    if not os.path.exists(user_bot_path):
        os.makedirs(user_bot_path)
    prompts_json_path = os.path.join(user_bot_path, "prompts.json")
    base_prompts_json_path = os.path.join(base_prompts_path, "prompts.json")
    if not os.path.exists(prompts_json_path) and os.path.exists(base_prompts_json_path):
        async with aiofiles.open(base_prompts_json_path, 'r') as base_file:
            base_prompts_content = await base_file.read()
        async with aiofiles.open(prompts_json_path, 'w') as user_file:
            await user_file.write(base_prompts_content)
    async with aiofiles.open(prompts_json_path, 'r') as file:
        prompts = json.loads(await file.read())
    main_prompt = prompts["main_prompt"].replace('<<NAME>>', bot_name)
    greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
    botnameupper = bot_name.upper()
    usernameupper = username.upper()
    collection_name = f"BOT_NAME_{bot_name}"
    main_conversation = MainConversation(username, user_id, bot_name, conv_length)
    while True:
        try:
            conversation_history = main_conversation.get_dict_conversation_history()
            con_hist = main_conversation.get_conversation_history()
            timestamp = timestamp_func()
            timestring = timestamp_to_datetime(timestamp)
            
            input_expansion = []
            input_expansion.append({'role': 'system', 'content': "You are a task rephraser. Your job is to take the user's latest input and rewrite it in a way that is clear and easy to understand. Only use the provided conversation history if it directly relates to the current input. Focus on making the current user input understandable on its own, without needing any previous conversation context. Just output the rewritten user input and nothing else."})
            input_expansion.append({'role': 'user', 'content': f"PREVIOUS CONVERSATION HISTORY (use only if directly relevant): {con_hist}\n\nCURRENT USER INPUT: {user_input}"})
            input_expansion.append({'role': 'assistant', 'content': f"EXPANDED INPUT: "})

            if len(conversation_history) > 1:
                if API == "OpenAi":
                    expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
                if API == "Oobabooga":
                    expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
                if API == "KoboldCpp":
                    expanded_input = await Input_Expansion_API_Call(API, backend_model, input_expansion, username, bot_name)
                if API == "AetherNode":
                    prompt = ''.join([message_dict['content'] for message_dict in input_expansion])
                    expanded_input = await Input_Expansion_API_Call(API, prompt, username, bot_name)
                expanded_input = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, expanded_input)
                if Debug_Output == "True":
                    print(f"\n\nEXPANDED INPUT: {expanded_input}\n\n")
            else:
                expanded_input = user_input
                
            def remove_duplicate_dicts(input_list):
                seen = {}
                for index, item in enumerate(input_list):
                    seen[item] = index 
                output_list = [item for item, idx in sorted(seen.items(), key=lambda x: x[1])]
                
                return output_list
  
            db_search_module_name = f'Resources.DB_Search.{vector_db}'
            db_search_module = importlib.import_module(db_search_module_name)
            client = db_search_module.initialize_client()

            domain_search = db_search_module.retrieve_domain_list(collection_name, bot_name, user_id, expanded_input, search_number=25)
            domain_selection = []    
            domain_selection.append({
                'role': 'system',
                'content': f"""You are a Domain Ontology Specialist. Your task is to match the given text or user query to one or more domains from the provided list. Follow these guidelines:

            1. Analyze the main subject(s) or topic(s) of the text.
            2. Select one or more domains from this exact list: {domain_search}
            3. Choose the domain(s) that best fit the main topic(s) of the text.
            4. You MUST only use domains from the provided list. Do not create or suggest new domains.
            5. Respond ONLY with the chosen domain name(s), separated by commas if multiple domains are selected.
            6. Do not include any explanations, comments, or additional punctuation.

            If no domain in the list closely matches the topic, choose the most relevant general category or categories from the list.

            Example domain list: ["Health", "Technology", "Finance", "Education"]

            Example input: "What are the benefits of regular exercise for cardiovascular health?"
            Example output: Health

            Example input: "How can artificial intelligence be used to improve online learning platforms?"
            Example output: Technology,Education

            Example input: "Discuss the impact of cryptocurrency on traditional banking systems."
            Example output: Finance,Technology"""
            })
            domain_selection.append({'role': 'user', 'content': f"Match this text to one or more domains from the provided list, separating multiple domains with commas: {expanded_input}"})
            domain_selection.append({'role': 'assistant', 'content': f"Domain: "})
            if API == "OpenAi":
                selected_domain = await Domain_Selection_API_Call(API, backend_model, domain_selection, username, bot_name)
            if API == "Oobabooga":
                selected_domain = await Domain_Selection_API_Call(API, backend_model, domain_selection, username, bot_name)
            if API == "KoboldCpp":
                selected_domain = await Domain_Selection_API_Call(API, backend_model, domain_selection, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in domain_selection])
                selected_domain = await Domain_Selection_API_Call(API, prompt, username, bot_name)
            selected_domain = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, selected_domain)
            if Debug_Output == "True":
                print(f"\n\nSELECTED DOMAIN: {selected_domain}\n\n")
                
            all_db_search_results = [] 
            domains = [domain.strip() for domain in re.split(r',\s*', selected_domain)]                     
            for domain in domains:
                tasklist = []          
                tasklist.append({'role': 'system', 'content': f"""SYSTEM: You are a search query coordinator. Your role is to interpret the original user query and generate 1-3 synonymous search terms that will guide the exploration of the chatbot's memory database. Each alternative term should reflect the essence of the user's initial search input within the context of the "{domain}" knowledge domain. Please list your results using bullet point format."""})
                tasklist.append({'role': 'user', 'content': f"USER: {user_input}\nADDITIONAL USER CONTEXT: {expanded_input}\nKNOWLEDGE DOMAIN: {domain}\n\nUse the format: •Search Query"})
                tasklist.append({'role': 'assistant', 'content': f"ASSISTANT: Sure, I'd be happy to help! Here are 1-3 synonymous search terms each starting with a '•': "})

                if API == "OpenAi":
                    tasklist_output = await Input_Expansion_API_Call(API, backend_model, tasklist, username, bot_name)
                if API == "Oobabooga":
                    tasklist_output = await Input_Expansion_API_Call(API, backend_model, tasklist, username, bot_name)
                if API == "KoboldCpp":
                    tasklist_output = await Input_Expansion_API_Call(API, backend_model, tasklist, username, bot_name)
                if API == "AetherNode":
                    prompt = ''.join([message_dict['content'] for message_dict in input_expansion])
                    tasklist_output = await Input_Expansion_API_Call(API, prompt, username, bot_name)
                tasklist_output = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, tasklist_output)
                
                if Debug_Output == 'True':
                    print(f"""\nSEMANTIC TERM SEPARATION FOR THE DOMAIN "{domain}":\n{tasklist_output}""")
                    
                lines = tasklist_output.splitlines()
                temp_db_result = []
                for line in lines:
                    domain_search = db_search_module.search_db(collection_name, bot_name, user_id, line, domain, search_number=15)
                    if len(domain_search) >= 5:
                        last_entries = domain_search[-5:]
                        domain_search = domain_search[:-5]
                    else:
                        last_entries = domain_search
                    
                    temp_db_result.extend(domain_search)
                    temp_db_result.extend(last_entries)

                all_db_search_results.extend(temp_db_result)
                temp_db_result.clear()
                
            all_db_search_results = remove_duplicate_dicts(all_db_search_results)
            
            limited_results = all_db_search_results[-35:]
            final_output = "\n".join([f"[ - {entry}]" for entry in limited_results])
            if Debug_Output == "True":
                print(f"\nORIGINAL DB SEARCH RESULTS:\n{final_output}")


            pruner = []
            pruner.append({'role': 'system', 'content': 
                """You are an article pruning assistant for a RAG (Retrieval-Augmented Generation) system. Your task is to filter out irrelevant articles and order the remaining ones by relevance, with the most relevant at the bottom."""})
            pruner.append({'role': 'assistant', 'content': 
                f"ARTICLES: {final_output}"})
            pruner.append({'role': 'user', 'content': 
                f"""QUESTION: {user_input}
                CONTEXT: {expanded_input}

                INSTRUCTIONS:
                1. Carefully read the question and context.
                2. Review all articles in the ARTICLES section.
                3. Remove only articles that are completely irrelevant to the question and context.
                4. Retain articles that have any level of relevance, even if it's indirect or provides background information.
                5. Arrange the remaining articles in order of relevance, with the most relevant at the bottom of the list.
                6. Copy selected articles EXACTLY, including their [- Article X] tags.
                7. Do not modify, summarize, or combine articles in any way.
                8. Separate articles with a blank line.
                9. When in doubt about an article's relevance, keep it.

                OUTPUT:
                Paste all remaining articles below, exactly as they appear in the original list, arranged with the most relevant at the bottom. Do not add any other text or explanations."""})
            pruner.append({'role': 'assistant', 'content': "PRUNED AND ARRANGED ARTICLES:\n\n"})
            if API == "OpenAi":
                pruned_entries = await DB_Prune_API_Call(API, backend_model, pruner, username, bot_name)
            if API == "Oobabooga":
                pruned_entries = await DB_Prune_API_Call(API, backend_model, pruner, username, bot_name)
            if API == "KoboldCpp":
                pruned_entries = await DB_Prune_API_Call(API, backend_model, pruner, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in pruner])
                pruned_entries = await DB_Prune_API_Call(API, prompt, username, bot_name) 
                
            if Debug_Output == "True":
                print(f"\nPRUNED ENTRIES:\n{pruned_entries}\n")  
                
            folder_path = "./Sub_Agents"
            folder_description_map = await load_folder_names_and_descriptions(folder_path)
            

            
            try:
                cat_set = set()
                cat_list = []
                for folder_name, description in folder_description_map.items():
                    cat = folder_name.upper()
                    cat_entry = f"[{folder_name}]- {description}"
                    print(f"\n\nCAT ENTRY: {cat_entry}\n\n")
                    if cat_entry not in cat_set:  
                        cat_set.add(cat_entry)
                        cat_list.append({'content': cat_entry})  
                
                subagent_cat = '\n'.join([message_dict['content'] for message_dict in cat_list])
            except Exception as e:
                traceback.print_exc()
                print(f"An error occurred: {e}")
                subagent_cat = "NO TOOLS AVAILABLE"
            
            intuition = []
            intuition.append({'role': 'system', 'content': f"""SYSTEM: Create a concise action plan for {bot_name} to develop a response to {username}'s most recent message. The action plan should be a single, detailed paragraph that covers the following steps:

            1. Identify the main question or purpose in {username}'s message.
            2. Select relevant information from RETURNED ENTRIES in internal databases.
            3. Extract important details from the PREVIOUS CONVERSATION HISTORY.
            4. Determine which AVAILABLE TOOLS can provide useful data.
            5. Specify the types of information needed, including internal and external sources.
            6. Outline the steps for using the relevant tools to gather data.
            7. Create a structured outline for the response.
            8. Plan for review and refinement of the response to ensure accuracy and clarity.

            Provide the action plan without referring to performing the steps, but focus on the specific information and actions needed to create a comprehensive response."""})
            intuition.append({'role': 'user', 'content': "Return entries related to the user's input for context. Ignore entries that are not relevant to the user's inquiry."})
            intuition.append({'role': 'assistant', 'content': f"RETURNED ENTRIES: {pruned_entries}"})
            intuition.append({'role': 'user', 'content': "Return the list of tools available for use."})
            intuition.append({'role': 'assistant', 'content': f"AVAILABLE TOOLS: {subagent_cat}"})
            intuition.append({'role': 'user', 'content': "Analyze and return the previous conversation history."})
            intuition.append({'role': 'assistant', 'content': f"PREVIOUS CONVERSATION HISTORY: {con_hist}"})
            intuition.append({'role': 'user', 'content': f"Now return the user's most recent inquiry."})
            intuition.append({'role': 'assistant', 'content': f"USER'S MOST RECENT MESSAGE: {user_input}\nADDITIONAL CONTEXT FOR USER MESSAGE: {expanded_input}"})
            intuition.append({'role': 'user', 'content': f"Now please provide a concise action plan in a single paragraph on how to best respond to the user's inquiry, covering all the steps mentioned in the system message."})
            intuition.append({'role': 'assistant', 'content': f"ACTION PLAN: "})

            if API == "OpenAi":
                intuition_response = await Intuition_API_Call(API, backend_model, intuition, username, bot_name)
            if API == "Oobabooga":
                intuition_response = await Intuition_API_Call(API, backend_model, intuition, username, bot_name)
            if API == "KoboldCpp":
                intuition_response = await Intuition_API_Call(API, backend_model, intuition, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in intuition])
                intuition_response = await Intuition_API_Call(API, prompt, username, bot_name) 
            intuition_response = format_responses(backend_model, assistant_input_start, assistant_input_end, botnameupper, intuition_response)
            print(f"\n\nACTION PLAN: {intuition_response}\n\n")
            intuition.clear()     
                
            master_tasklist = []    
            master_tasklist.append({'role': 'system', 'content': f"MAIN SYSTEM PROMPT: You are a task list coordinator for {bot_name}, an autonomous AI chatbot. Your job is to create a list of 3-6 specific, independent research tasks based on the user's input and the user-facing chatbot's action plan. Each task should be assigned a specific Tool Category from the provided list, formatted as '[GIVEN CATEGORY]'. Do not alter or create new categories. The tasks will be executed by separate AI agents in a cluster computing environment, which are stateless and cannot communicate with each other or the user during task execution. Ensure that tools for searching internal data are prioritized for informational tasks unless the task explicitly requires real-time data. The goal is to provide a verbose, factually accurate, and comprehensive response. Avoid tasks focused on final product production, user communication, seeking external help, seeking external validation, or liaising with other entities. Respond using the following format: '•[GIVEN CATEGORY]: <TASK>\n•[GIVEN CATEGORY2]: <TASK2>'.\n\nNow, please return the Chatbot's Action Plan and available Tool List."})
            master_tasklist.append({'role': 'user', 'content': f"USER FACING CHATBOT'S ACTION PLAN: {intuition_response}\n\nAVAILABLE TOOL CATEGORIES:\n{subagent_cat}"})
            master_tasklist.append({'role': 'user', 'content': f"USER INQUIRY: {user_input}\nUse only the given categories from the provided list. Do not create or use any categories outside of the given Tool Categories. Prioritize internal databases for informational tasks unless real-time data is specifically required."})
            master_tasklist.append({'role': 'assistant', 'content': f"TASK COORDINATOR: Sure, here is a bullet point list of 3-6 tasks, each strictly assigned a category from the given Tool Categories to provide a verbose, factually accurate, and comprehensive response: "})

            if API == "OpenAi":
                master_tasklist_output = await Intuition_API_Call(API, backend_model, master_tasklist, username, bot_name)
            if API == "Oobabooga":
                master_tasklist_output = await Intuition_API_Call(API, backend_model, master_tasklist, username, bot_name)
            if API == "KoboldCpp":
                master_tasklist_output = await Intuition_API_Call(API, backend_model, master_tasklist, username, bot_name)
            if API == "AetherNode":
                prompt = ''.join([message_dict['content'] for message_dict in master_tasklist])
                master_tasklist_output = await Intuition_API_Call(API, prompt, username, bot_name) 
                  
            print(f"\n\nTASKLIST OUTPUT:\n{master_tasklist_output}")
            master_tasklist.clear()
            
            try:
                with open('Settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    
                if API == "Oobabooga":
                    host_data = settings.get('HOST_Oobabooga', 'http://localhost:5000/api').strip()
                if API == "AetherNode":
                    host_data = settings.get('HOST_AetherNode', 'http://127.0.0.1:8000').strip()
                if API == "KoboldCpp":
                    host_data = settings.get('HOST_KoboldCpp', 'http://127.0.0.1:5001').strip()
                if API == "OpenAi":
                    host_data = f"No Host for API"
                hosts = host_data.split(' ')
                num_hosts = len(hosts)
            except Exception as e:
                print(f"An error occurred while reading the host file: {e}") 
                
            host_queue = asyncio.Queue()
            for host in hosts:
                await host_queue.put(host)
            try:
                lines = re.split(r'\n\s*•\s*|\n\n', master_tasklist_output)
                lines = [line.strip() for line in lines if line.strip()]
            except Exception as e:
                print(f"An error occurred: {e}")
                lines = [master_tasklist_output]
                
                   
            task = {}
            task_result = {}
            task_result2 = {}
            task_counter = 1
                
            tasklist_completion = []    
            
            
            new_prompt = """You are {bot_name}, an AI assistant designed to answer {username}'s questions using only the provided context and the results of completed tasks. Follow these guidelines carefully:

            1. Use ONLY the information from the given context window and the insights gathered from the completed tasks for your responses.
            2. If the required information is not in the context or task results, respond with: "INFORMATION NOT FOUND IN DATABASE."
            3. Do not use any external knowledge or information outside the provided context and task results.
            4. Consider all information not present in the context window as outdated or potentially incorrect.
            5. The context entries are formatted as follows:
               [- Entry 1]
               [- Entry 2]
               [- Entry 3]

            6. Always refer to the context and task results before formulating your answer.
            7. Provide concise, relevant, and factually accurate answers based solely on the context and completed tasks.
            8. If multiple relevant entries or task results exist, synthesize the information coherently.
            9. Maintain the persona of {bot_name} in your responses.
            10. If asked about your capabilities, refer only to what's possible with the given context and task results.

            Remember: Your primary function is to provide accurate information from the context window and completed tasks. Do not speculate or infer beyond what is explicitly stated."""

            
            tasklist_completion.append({'role': 'system', 'content': f"MAIN SYSTEM PROMPT: {main_prompt}\n{new_prompt}"})
            tasklist_completion.append({'role': 'assistant', 'content': f"You are the final response module for the cluster compute Ai-Chatbot {bot_name}. Your job is to take the completed task list, and then give a verbose response to the end user in accordance with their initial request."})
            tasklist_completion.append({'role': 'user', 'content': f"FULL TASKLIST: {master_tasklist_output}"})
                
            print("\n\n----------------------------------\n\n")
            try:
                tasks = []
                for task_counter, line in enumerate(lines, start=1):
                    if line != "None":
                        task = asyncio.create_task(
                            wrapped_process_line(
                                host_queue, bot_name, username, line, task_counter, 
                                expanded_input, intuition_response, master_tasklist_output, 
                                user_input, folder_description_map, subagent_cat, user_id
                            )
                        )
                        tasks.append(task)
                completed_tasks = await asyncio.gather(*tasks)
                for task_result in completed_tasks:
                    tasklist_completion.extend(task_result)
            except Exception as e:
                print(f"An error occurred while executing tasks: {e}")
              
                
            try:            
                tasklist_completion.append({'role': 'assistant', 'content': f"USER'S INITIAL INPUT: {user_input}"})
                tasklist_completion.append({'role': 'user', 'content': f"SYSTEM: You are tasked with crafting a comprehensive, factually accurate, response for {username}. Use the insights and information gathered from the completed tasks during the research task loop to formulate your answer and provide factual backing. Since {username} does not have access to the research process, ensure that your reply is self-contained, providing all necessary context and information. Do not introduce information beyond what was discovered during the research tasks, and ensure that factual accuracy is maintained throughout your response.  If the needed information is not contained within the completed tasks, print: 'INFORMATION NOT FOUND IN DATABASE'\n\nUSER'S INITIAL INPUT: {user_input}\nYour research and planning phase is concluded. Concentrate on composing a detailed, coherent, and conversational reply that fully addresses the user's question based on the completed research tasks. Remember, the user cannot see any of the research and information must be reiterated."})
                tasklist_completion.append({'role': 'assistant', 'content': f"{botnameupper}: Here is my final response to the end-user's initial input: "})

                if API == "AetherNode":
                    prompt = ''.join([message_dict['content'] for message_dict in tasklist_completion])
                    final_response = await LLM_API_Call(API, backend_model, tasklist_completion, username, bot_name)
                if API == "OpenAi":
                    final_response = await LLM_API_Call(API, backend_model, tasklist_completion, username, bot_name)
                if API == "KoboldCpp":
                    final_response = await LLM_API_Call(API, backend_model, tasklist_completion, username, bot_name)
                if API == "Oobabooga":
                    final_response = await LLM_API_Call(API, backend_model, tasklist_completion, username, bot_name)
                print(f"\n\nFINAL RESPONSE: {final_response}\n\n")
            except Exception as e:
                traceback.print_exc()
                print(f"An error occurred: {e}")
                
            
            context_check = f"{domain_search}"
            dataset = []
            llama_3 = "Llama_3"
            heuristic_input_start2, heuristic_input_end2, system_input_start2, system_input_end2, user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2 = set_format_variables(Dataset_Format)
            formated_conversation_history = main_conversation.get_dict_formated_conversation_history(user_input_start2, user_input_end2, assistant_input_start2, assistant_input_end2)

            if len(context_check) > 10:
                dataset_prompt_1 = f"Here is your context window for factual verification, use any information contained inside over latent knowledge.\nCONTEXT WINDOW: [{domain_search}]"
                dataset_prompt_2 = f"Thank you for providing the context window, please now provide the conversation with the user."
                dataset.append({'role': 'user', 'content': f"{user_input_start2}{dataset_prompt_1}{user_input_end2}"})
                dataset.append({'role': 'assistant', 'content': f"{assistant_input_start2}{dataset_prompt_2}{assistant_input_end2}"})
                dataset.append({'role': 'user', 'content': f"I will now provide the previous conversation history:"})
                
            if len(formated_conversation_history) > 1:
                if len(greeting_msg) > 1:
                    dataset.append({'role': 'assistant', 'content': f"{greeting_msg}"})
                for entry in formated_conversation_history:
                    dataset.append(entry)

            dataset.append({'role': 'user', 'content': f"{user_input_start2}{user_input}{user_input_end2}"})
            filtered_content = [entry['content'] for entry in dataset if entry['role'] in ['user', 'assistant']]
            llm_input = '\n'.join(filtered_content)
            heuristic = f"{heuristic_input_start2}{main_prompt}{heuristic_input_end2}"
            system_prompt = f"{system_input_start2}{new_prompt}{system_input_end2}"
            assistant_response = f"{assistant_input_start2}{final_response}{assistant_input_end2}"
            if Dataset_Output == 'True':
                print(f"\n\nHEURISTIC: {heuristic}")
                print(f"\n\nSYSTEM PROMPT: {system_prompt}")
                print(f"\n\nINPUT: {llm_input}")  
                print(f"\n\nRESPONSE: {assistant_response}")
                     
            if Write_Dataset == 'True':
                print(f"\n\nWould you like to write to dataset? Y or N?")   
                while True:
                    try:
                        yorno = input().strip().upper() 
                        if yorno == 'Y':
                            print(f"\n\nWould you like to include the conversation history? Y or N?")
                            while True:
                                yorno2 = input().strip().upper() 
                                if yorno2 == 'Y':
                                    if Dataset_Upload_Type == 'Custom':
                                        write_dataset_custom(Dataset_Format, heuristic, system_prompt, llm_input, assistant_response)
                                        print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                                    if Dataset_Upload_Type == 'Simple':
                                        write_dataset_simple(Dataset_Format, llm_input, final_response)
                                        print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
                                    break  
                                elif yorno2 == 'N':
                                    if Dataset_Upload_Type == 'Custom':
                                        write_dataset_custom(Dataset_Format, heuristic, system_prompt, user_input, assistant_response)
                                        print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                                    if Dataset_Upload_Type == 'Simple':
                                        write_dataset_simple(Dataset_Format, user_input, final_response)
                                        print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
                                    break 
                                else:
                                    print("Invalid input. Please enter 'Y' or 'N'.")

                            break  
                        elif yorno == 'N':
                            print("Not written to Dataset.\n\n")
                            break 
                        else:
                            print("Invalid input. Please enter 'Y' or 'N'.")
                    except:
                        traceback.print_exc()
            if Write_Dataset == 'Auto':
                if Dataset_Upload_Type == 'Custom':
                    write_dataset_custom(Dataset_Format, heuristic, system_prompt, llm_input, assistant_response)
                    print(f"Written to {Dataset_Format}_custom_dataset.json\n\n")
                if Dataset_Upload_Type == 'Simple':
                    write_dataset_simple(Dataset_Format, user_input, final_response)
                    print(f"Written to {Dataset_Format}_simple_dataset.json\n\n")
            main_conversation.append(timestring, user_input, final_response)
            if Debug_Output == 'True':
                print("\n\n\n")
            return heuristic, system_prompt, llm_input, user_input, final_response
        except:
            error = traceback.print_exc()
            error1 = traceback.print_exc()
            error2 = traceback.print_exc()
            error3 = traceback.print_exc()
            error4 = traceback.print_exc()
            return error, error1, error2, error3, error4
            
            
            
async def wrapped_process_line(host_queue, bot_name, username, line, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input, folder_description_map, subagent_cat, user_id):
    host = await host_queue.get()
    result = await process_line(host, host_queue, bot_name, username, line, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input, folder_description_map, subagent_cat, user_id)
    await host_queue.put(host)
    return result   
                
                
async def process_line(host, host_queue, bot_name, username, line, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input, folder_description_map, subagent_cat, user_id):
    tasklist_completion2 = list()
    conversation = list()
    cat_list = list()
    cat_choose = list()
    try:
        with open('./Settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        API = settings.get('API', 'AetherNode')

            
        Sub_Module_Output = settings.get('Output_Sub_Module', 'False')
        completed_task = "Error Completing Task"
        backend_model = settings.get('Model_Backend', 'Llama_2_Chat')
        select_api = settings.get('API', 'Oobabooga')
        Use_Char_Card = settings.get('Use_Character_Card', 'False')
        Char_File_Name = settings.get('Character_Card_File_Name', 'Aetherius')
        Write_Dataset = settings.get('Write_To_Dataset', 'False')
        Dataset_Upload_Type = settings.get('Dataset_Upload_Type', 'Custom')
        Dataset_Format = settings.get('Dataset_Format', 'Llama_3')
        LLM_API_Call, Agent_LLM_API_Call, Input_Expansion_API_Call, Domain_Selection_API_Call, DB_Prune_API_Call, Inner_Monologue_API_Call, Intuition_API_Call, Final_Response_API_Call, Short_Term_Memory_API_Call = import_api_function()
        heuristic_input_start, heuristic_input_end, system_input_start, system_input_end, user_input_start, user_input_end, assistant_input_start, assistant_input_end = set_format_variables(backend_model)
        end_prompt = ""
        botnameupper = bot_name.upper()
        usernameupper = username.upper()
        try:
            for folder_name in folder_description_map.keys():
                cat = folder_name.upper()
                cat_list.append(cat)

            category_found = False
            lineupper = line.upper()
            content_within_brackets = re.findall(r'\[(.*?)\]', lineupper)
            for cat in cat_list:
                for content in content_within_brackets:
                    if cat in content:
                        line_cat = cat
                        category_found = True
                        break
                                            
            if not category_found:
                cat_choose.append({'role': 'system', 'content': f"ROLE: You are functioning as a sub-module within a category selection tool. OBJECTIVE: Your primary responsibility is to reevaluate and reassign categories. If a task is associated with a category not present in the predefined list, you must reassign it to an applicable existing category from the list. FORMAT: Maintain the original task wording, and follow this structure for the reassigned task: '[NEW CATEGORY]: <TASK>'. Upon completion, return the task with the updated category assignment."})
                cat_choose.append({'role': 'assistant', 'content': f"AVAILABLE TOOL CATEGORIES: {subagent_cat}"})
                cat_choose.append({'role': 'user', 'content': f"TASK REQUIRING CATEGORY REASSIGNMENT: {line}"})
                   
                if API == "OpenAi":
                    task_expansion = await Agent_LLM_API_Call(API, backend_model, cat_choose, username, bot_name)
                if API == "Oobabooga":
                    task_expansion = await Agent_LLM_API_Call(host, API, backend_model, cat_choose, username, bot_name)
                if API == "KoboldCpp":
                    task_expansion = await Agent_LLM_API_Call(host, API, backend_model, cat_choose, username, bot_name)
                if API == "AetherNode":
                    prompt = ''.join([message_dict['content'] for message_dict in cat_choose])
                    task_expansion = await Agent_LLM_API_Call(host, API, prompt, username, bot_name)
                    
                task_expansion = task_expansion.upper()
                category_matches = re.findall(r'\[(.*?)\]', task_expansion)
                for cat in cat_list:
                    for matched_category in category_matches:
                        if cat.upper() in matched_category.upper():
                            line_cat = matched_category
                            category_found = True
                            break
                    if category_found:
                        break
        except Exception:
            traceback_print_out = traceback.print_exc()
            print(f"An error occurred: {traceback_print_out}") 
            
        if category_found:
            sub_agents_path = './Sub_Agents'
            best_match = None
            best_match_count = 0

            for folder in os.listdir(sub_agents_path):
                if os.path.isdir(os.path.join(sub_agents_path, folder)):
                    match_count = sum(1 for c in folder.upper() if c in line_cat.upper())
                    if match_count > best_match_count:
                        best_match = folder
                        best_match_count = match_count
            subagent_list = [] 
            if best_match:
                tools_path = os.path.join(sub_agents_path, best_match)
                
                for tool in os.listdir(tools_path):
                    if tool.endswith('.py'):
                        tool_name = tool[:-3] 
                        tool_module_path = os.path.join(tools_path, tool)
                        with open(tool_module_path, 'r') as f:
                            code = f.read()
                            exec(code) 
                        description_func_name = f"{tool_name}_Description"
                        if description_func_name in locals():
                            description = locals()[description_func_name](username, bot_name)
                            subagent_list.append({'name': tool_name, 'description': description})
                        else:
                            print(f"No description function found for tool: {tool_name}")
            else:
                print(f"No folder found with sufficient match for category: {line_cat}")
            
        tasklist_completion2.append({'role': 'user', 'content': f"TASK: {line}"})
        
        conversation.append({'role': 'assistant', 'content': f"First, please query your tool database to identify the tools that are currently available to you. Remember, you can only use these tools."})
        conversation.append({'role': 'assistant', 'content': f"AVAILABLE TOOLS: {subagent_list} "})
        conversation.append({'role': 'user', 'content': f"Your task is to select one of the given tools to complete the assigned task from the provided list of available tools. Ensure that your choice is strictly based on the options provided, and do not suggest or introduce tools that are not part of the list. Your response should be a concise answer that distinctly identifies the chosen tool without going into the operational process or detailed usage of the tool."})
        conversation.append({'role': 'assistant', 'content': f"ASSIGNED TASK: {line}"})
        
        if API == "OpenAi":
            task_expansion = await Agent_LLM_API_Call(API, backend_model, conversation, username, bot_name)
        if API == "Oobabooga":
            task_expansion = await Agent_LLM_API_Call(host, API, backend_model, conversation, username, bot_name)
        if API == "KoboldCpp":
            task_expansion = await Agent_LLM_API_Call(host, API, backend_model, conversation, username, bot_name)
        if API == "AetherNode":
            prompt = ''.join([message_dict['content'] for message_dict in conversation])
            task_expansion = await Agent_LLM_API_Call(host, API, prompt, username, bot_name)
            
                
        best_match = None
        best_match_score = 0
        
        for subagent in subagent_list:
            tool_name = subagent['name']
            match_score = sum(1 for c in tool_name.lower() if c in task_expansion.lower())
            
            # Update if this tool has a better match score
            if match_score > best_match_score:
                best_match = tool_name
                best_match_score = match_score
        
        if best_match:
            subagent_selection = [f"{best_match}.py"]
        else:
            subagent_selection = "No Sub-Agents Found"
                    
        if subagent_selection != "No Sub-Agents Found":
            tasks = []  
            if not subagent_selection:
                print("\nError with Module, using fallback")
                
                fallback_path = ".\Sub_Agents\Factual_Backing\Database_Search.py"
                subagent_selection = [os.path.basename(fallback_path)]
                
            for filename_with_extension in subagent_selection:
                filename = filename_with_extension.rstrip('.py')
                script_path = os.path.join(f'.\Sub_Agents\{line_cat}', filename_with_extension)
                if os.path.exists(script_path):
                    spec = spec_from_file_location(filename, script_path)
                    module = module_from_spec(spec)
                    spec.loader.exec_module(module)
                    function_to_call = getattr(module, filename, None)    
                    if function_to_call is not None:
                        if asyncio.iscoroutinefunction(function_to_call):
                            task = function_to_call(host, bot_name, username, user_id, line, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input)
                        else:
                            loop = asyncio.get_running_loop()
                            task = loop.run_in_executor(None, function_to_call, host, bot_name, username, user_id, line, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input)
                        tasks.append(task)
            completed_task = await asyncio.gather(*tasks)
            tasklist_completion2.append({'role': 'assistant', 'content': f"SUB-AGENT {task_counter} OUTPUT: {completed_task} "})
            print(f"CATEGORY: {line_cat}\n")
            print(f"SELECTED SUB-AGENT: {subagent_selection}\n")
            print(f"SUB-AGENT {task_counter} OUTPUT: {completed_task}")
            print("\n\n----------------------------------\n\n")
        return tasklist_completion2
    except Exception:
        traceback_print_out = traceback.print_exc()
        print(f"An error occurred: {traceback_print_out}")
        tasklist_completion2.append({'role': 'assistant', 'content': f"ERROR WHEN COMPLETING TASK: {traceback_print_out}"})
        return tasklist_completion2   
                     

async def main():
    print("\n")
    print(f"If you found this example useful, please give it a star and consider tipping me on Ko-fi :)")
    print("\n\n")
    with open('./Settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    username = settings.get('Username', 'User')
    user_id = settings.get('User_ID', 'UNIQUE_USER_ID')
    bot_name = settings.get('Bot_Name', 'Chatbot')
    conv_length = settings.get('Conversation_Length', '3')
    history = []
    base_path = "./Chatbot_Prompts"
    base_prompts_path = os.path.join(base_path, "Base")
    user_bot_path = os.path.join(base_path, user_id, bot_name)  
    if not os.path.exists(user_bot_path):
        os.makedirs(user_bot_path)
    prompts_json_path = os.path.join(user_bot_path, "prompts.json")
    base_prompts_json_path = os.path.join(base_prompts_path, "prompts.json")
    if not os.path.exists(prompts_json_path) and os.path.exists(base_prompts_json_path):
        async with aiofiles.open(base_prompts_json_path, 'r') as base_file:
            base_prompts_content = await base_file.read()
        async with aiofiles.open(prompts_json_path, 'w') as user_file:
            await user_file.write(base_prompts_content)
    async with aiofiles.open(prompts_json_path, 'r') as file:
        prompts = json.loads(await file.read())
    greeting_msg = prompts["greeting_prompt"].replace('<<NAME>>', bot_name)
    while True:
        main_conversation = MainConversation(username, user_id, bot_name, conv_length)
        conversation_history = main_conversation.get_dict_conversation_history()
        con_hist = main_conversation.get_conversation_history()
        print(con_hist)
        if len(conversation_history) < 1:
            print(f"{bot_name}: {greeting_msg}\n")
        user_input = input(f"{username}: ")
        if user_input.lower() == 'exit':
            break
        response = await Agentic_RAG_Chatbot(user_input, username, user_id, bot_name)
        history.append({"user": user_input, "bot": response})

if __name__ == "__main__":
    asyncio.run(main())