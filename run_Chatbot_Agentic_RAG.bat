@echo off
cd /d "%~dp0"
call venv\Scripts\activate


echo Running the project...
python Chatbot_Agentic_RAG.py

echo Press any key to exit...
pause >nul