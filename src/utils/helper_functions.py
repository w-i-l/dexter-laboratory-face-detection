import os

def format_path(path: str) -> str:
    if path == None:
        return None
    
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
     
    return path