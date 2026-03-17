import os

# Maximum characters allowed for certain string operations to prevent memory issues
MAX_CHARS = 10000

def get_file_content(working_directory: str, file_path: str, start_line: int | None = None, end_line: int | None = None):
    abs_working_directory = os.path.abspath(working_directory)
    abs_file_path = os.path.abspath(os.path.join(working_directory, file_path))

    # Prevent the agent from trying to read system files outside the workspace
    if not abs_file_path.startswith(abs_working_directory):
        return f'Error: "{file_path}" is not in the working directory.' 

    # Verify the file actually exists on disk
    if not os.path.isfile(abs_file_path):
        return f'Error: "{file_path}" is not a valid file.'
    

    file_content_string = ""

    try: 
        # Open in read mode
        with open(abs_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        
        # Handle 1-indexed line parameters
        start_idx = max(0, start_line - 1) if start_line is not None else 0
        end_idx = min(total_lines, end_line) if end_line is not None else total_lines
        
        sliced_lines = lines[start_idx:end_idx]
        
        # Prepend line numbers so the LLM knows exactly where it is (useful for edit_file)
        numbered_lines = []
        for i, line in enumerate(sliced_lines):
            actual_line_num = start_idx + i + 1
            numbered_lines.append(f"{actual_line_num:4d} | {line}")
            
        file_content_string = "".join(numbered_lines)

        # Only read up to the maximum characters limit to avoid massive context bloating
        if len(file_content_string) >= MAX_CHARS:
            file_content_string = file_content_string[:MAX_CHARS]
            file_content_string += (
                f'\n\n[...File "{file_path}" truncated at {MAX_CHARS} characters. '
                f'Use start_line and end_line to read the rest.]'
            )

        return file_content_string
    
    except Exception as e:
        return f"Exception reading file: {e}"