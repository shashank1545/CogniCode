import os
import subprocess
from langchain.tools import tool
import pathspec

from rag import retrieval, context_manager



@tool
def list_files(directory: str = '.', use_gitignore: bool = True):
    """
    Recursively lists files and subdirectories within a given directory.

    **Primary Use Case:**
    - To explore and understand the structure of a directory when you don't know its contents.
    - To get a "map" of a part of the project before deciding which files to read or modify.

    **When to use this tool:**
    - When the user asks an open-ended question about what is in a directory (e.g., "What files are in the 'src' folder?").
    - As a preliminary step before using `read_file`.

    **IMPORTANT:** Do NOT use this tool if the user's question can be answered more directly by `run_shell_command`. For example, to count files, use `run_shell_command` with `find` and `wc`, as it is much more efficient.
    """

    if not os.path.isdir(directory):
        return f"Error: Directory {directory} not found."

    all_files = []
    
    # Load .gitignore patterns
    spec = None
    if use_gitignore:
        gitignore_path = os.path.join(directory, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                spec = pathspec.PathSpec.from_lines('gitwildmatch', f)

    for root, dirs, files in os.walk(directory, topdown=True):
        # Always ignore .git directory
        if '.git' in dirs:
            dirs.remove('.git')

        # Paths should be relative to the directory where the gitignore is located for matching
        relative_root = os.path.relpath(root, directory)
        if relative_root == '.':
            relative_root = ''

        if spec:
            # Filter directories
            paths_to_check = [os.path.join(relative_root, d) for d in dirs]
            ignored_dirs = set(spec.match_files(paths_to_check))
            
            # Reconstruct ignored dir names from paths
            ignored_dir_names = set()
            for ignored_path in ignored_dirs:
                # This is a simplification. It will not work correctly if the pattern has slashes.
                ignored_dir_names.add(os.path.basename(ignored_path))

            dirs[:] = [d for d in dirs if d not in ignored_dir_names]

            # Filter files
            paths_to_check = [os.path.join(relative_root, f) for f in files]
            ignored_files = set(spec.match_files(paths_to_check))
            
            ignored_file_names = set()
            for ignored_path in ignored_files:
                ignored_file_names.add(os.path.basename(ignored_path))

            for name in files:
                if name not in ignored_file_names:
                    all_files.append(os.path.join(root, name))
        else:
            for name in files:
                all_files.append(os.path.join(root, name))


    if not all_files:
        return f"The directory {directory} is empty."

    return "\n".join(all_files)

@tool
def read_file(file_path: str):
    """
    Reads and returns the full content of a single specified file.

    **Primary Use Case:**
    - To inspect the code or content of a specific file when you know its path.
    - To understand what a file does before modifying it.

    **When to use this tool:**
    - When you have a file path and need to see what's inside.
    - After using `list_files` or `run_shell_command` (with `find`) to locate a file of interest.

    **IMPORTANT:** This tool is for reading one file at a time. Do not use it if another tool like `run_shell_command` (e.g., with `cat` or `grep`) could be more efficient for the user's goal.
    """

    try:
        with open(file_path, "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"


@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command and returns its output. This is a powerful and versatile tool that should be your default choice for a wide range of tasks.

    **Primary Use Cases:**
    - **File Operations:** Counting files (e.g., `find . -name '*.py' | wc -l`), finding files (`find . -name 'config.py'`), checking file sizes (`du -sh .`).
    - **System Information:** Checking git status (`git status`), viewing process lists (`ps aux`), checking disk space (`df -h`).
    - **Running Scripts & Tests:** Executing scripts (`python my_script.py`), running test suites (`pytest`).

    **When to use this tool:**
    - When a direct command-line action can precisely and efficiently answer the user's question.
    - For any task related to system state, file manipulation, or running external processes.
    - If you are unsure which tool to use, consider if a shell command could solve the problem. This is often the most powerful tool available.

    **Input:** A valid shell command string.
    **Output:** The stdout and stderr from the command.
    """
    try:
        result = subprocess.run(
        command,
        shell= True,
        capture_output=True,
        text=True,
        check=True
    )
        output = f"\nSTDOUT:{result.stdout}\n"
        if result.stderr:
            output += f"\nSTDERR:{result.stderr}\n"
        return output

    except subprocess.CalledProcessError as e:
        return f"""\nSTDERR:Error executing command: '{command}'\nExit Code:{e.returncode}
                \nSTDOUT:\n {e.stdout}\nSTDERR\n{e.stderr}"""
    except Exception as e:
        return f"\nSTDERR:An unexpected error occurred : {e}"


@tool
def codebase_search(query: str)-> str:
    """
    Performs a semantic search over the codebase using a RAG (Retrieval-Augmented Generation) pipeline.

    **Primary Use Case:**
    - To find relevant code snippets or concepts when you do not know the exact file name or location.
    - To answer broad, conceptual questions about the codebase (e.g., "How is user authentication handled?").

    **When to use this tool:**

    **IMPORTANT:** This tool returns retrieved context, not a final answer. You must synthesize the context to answer the user's question. For simple file finding, `run_shell_command` with `find` is often more direct.
    """

    print(f"Agent is using codebase_search tool with query :'{query}'")

    retrieved_docs = retrieval.hybrid_search(query, k=10)

    context_str = context_manager.build_context(
        retrieved_docs,
        top_k=3,
        map_reduce_count=5
    )

    if not context_str:
        return "No relevant information found in codebase for the query."

    return f"\nCONTEXT:{context_str}\n"
