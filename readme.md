# Agentic RAG
Version 0.01 of Agentic RAG Example by [LibraryofCelsus.com](https://www.libraryofcelsus.com)  
  
[Installation Guide](#installation-guide)  
[Skip to Changelog](#changelog)  
[Discord Server](https://discord.gg/pb5zcNa7zE)

------
**Recent Changes**

• 08/09 First Release

------

### What is this project?

This repository demonstrates a Retrieval-Augmented Generation (RAG) technique using Sub-Agents to better currate needed data. This technique was originally developed for use with my Aetherius Ai Assistant Project. This standalone example will be used to improve and test the technique before reintegration with the main project.

This project serves as a chatbot component for: https://github.com/libraryofcelsus/LLM_File_Parser

Main Ai Assistant Github: https://github.com/libraryofcelsus/Aetherius_AI_Assistant  

------

My Ai work is self-funded by my day job, consider supporting me if you appreciate my work.

<a href='https://ko-fi.com/libraryofcelsus' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

------

Join the Discord for help or to get more in-depth information!

Discord Server: https://discord.gg/pb5zcNa7zE

Made by: https://github.com/libraryofcelsus


------
# Future Plans: 
• Add more Sub-Agents  
• Improve Internal Prompts  

# Changelog: 
**0.01** 

• First Release

# How to create a custom Sub-Agent:
The sub-agent system functions with a category system.  The framework will first select a Category for the generated task, then select a Sub-Agent from within the Category.

To create a category simply Create a folder within the ./Sub_Agents folder with the desired category name.  Inside of the folder create a .txt file with a description of what kind of tasks the category should be used for.  The name of the txt file should be the same as the Category name.

Inside of the folder is where you will place your actual Sub-Agent script.  Each Agent needs two functions.  
The first is a description function that will be used to select what agent should be used for the task.  The function's name should be "{file_name}_Descritpion".    

The second function needed is the actual Sub_Agent script.  This function should just be named "{file_name}".
The variables that are passed through to the script are as follows:  
**(host, bot_name, username, user_id, task, task_counter, expanded_input, intuition_response, master_tasklist_output, user_input)**   
**host** = The host that will be used for API generation.  This is meant to be used with imported LLM calls to increase generation.  It can be ignored if you are using OpenAi or a single machine.   
**bot_name** = The name of the main Chatbot.   
**username** = The name of the current User.   
**user_id** = Unique identifier for the User.     
**task** = The current generated task that is passed onto the sub-agent.  
**task_counter** = A number representitive of the current task.  
**expanded_input** = The user's input expanded with additional context from the conversation history.   
**intuition_response** = Generated Action Plan for how the Chatbot should give a final response.  
**master_tasklist_output** = Complete list of generated tasks for the main chatbot.  
**user_input** = The original User Input.   

From there, you can create a script that completes the task however you see fit.  After a response to the task is generated, it must be returned using OpenAi formatting.  
Example:  
        sub_agent_completion.append({'role': 'user', 'content': f"TASK {task_counter}: {line}"})   
        sub_agent_completion.append({'role': 'assistant', 'content': f"COMPLETED TASK {task_counter}: {task_completion}"})   
        return sub_agent_completion   



# Installation Guide

## Installer bat

Download the project zip folder by pressing the <> Code drop down menu.

**Note: Project uses OpenAi and Qdrant by default.  This can be changed in ./Settings.json**

**1.** Install Python 3.10.6, Make sure you add it to PATH: **https://www.python.org/downloads/release/python-3106/**

**2.** Run "install_requirements.bat" to install the needed dependencies.  The bat will install Git, and the needed python dependencies.  

(If you get an error when installing requirements run: **python -m pip cache purge**)

**3.** Set up Qdrant or Marqo DB.  To change what DB is used, edit the "Vector_DB" Key in ./Settings.json.  Qdrant is the default. 

Qdrant Docs: https://qdrant.tech/documentation/guides/installation/   

Marqo Docs: https://docs.marqo.ai/2.9/  

To use a local Qdrant server, first install Docker: https://www.docker.com.  
Next type: **docker pull qdrant/qdrant:v1.9.1** in the command prompt.  
After it is finished downloading, type **docker run -p 6333:6333 qdrant/qdrant:v1.9.1**  

To use a local Marqo server, first install Docker: https://www.docker.com.  
Next type: **docker pull marqoai/marqo:latest** in the command prompt.  
After it is finished downloading, type **docker run --name marqo --gpus all -p 8882:8882 marqoai/marqo:latest**   

(If it gives an error, check the docker containers tab for a new container and press the start button.  Sometimes it fails to start.)  

See: https://docs.docker.com/desktop/backup-and-restore/ for how to make backups.  

Once the local Vector DB server is running, it should be auto detected by the scripts.   

**6.** Install your desired API or enter OpenAi Key in ./Settings.json.  To change what Api is used, edit the "API" Key in ./Settings.json  
https://github.com/oobabooga/text-generation-webui  

Models tested with:   
• Meta-Llama-3-8B-Instruct-Q6_K.gguf   

**8.** Launch a script with one of the **run_*.bat**  

**9.** Change the information inside of the "Settings" tab to your preferences.  

**10.** Start chatting with your data!  Data can be uploaded from your own files using:  https://github.com/libraryofcelsus/LLM_File_Parser




-----

# Contact
Discord: libraryofcelsus      -> Old Username Style: Celsus#0262

MEGA Chat: https://mega.nz/C!pmNmEIZQ