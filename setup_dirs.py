import os

# Create directory structure
dirs = [
    'src',
    'src/utils',
    'src/analyzers',
    'src/cleaners',
    'src/ui'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
