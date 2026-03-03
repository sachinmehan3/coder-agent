import os

# Maximum characters allowed for certain string operations to prevent memory issues
MAX_CHARS = 10000

def get_file_content(working_directory, file_path):
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
            # Only read up to the maximum characters limit to avoid massive context bloating
            file_content_string = f.read(MAX_CHARS)

            # Inform the AI if the file was truncated so it is aware it's only seeing part of it
            if len(file_content_string) >= MAX_CHARS:
                file_content_string += (
                    f'[...File "{file_path}" truncated at 10000 characters]'
                    )


        return file_content_string
    
    except Exception as e:
        return f"Exception reading file: {e}"