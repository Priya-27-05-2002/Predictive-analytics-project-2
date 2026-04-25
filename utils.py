import os

def create_directories():
    """Create the necessary directory structure for the project workflow."""
    dirs = ['data', 'models', 'notebooks', 'src']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
