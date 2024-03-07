import os
from tree_sitter import Language, Parser
 # Define the path to the 'languages' folder
languages_folder = os.path.join("dependencies","languages")
language =  "java"
# Check if the 'python' folder exists inside the 'languages' folder
folder_path = os.path.join(languages_folder, language)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
Language.build_library(
    os.path.join(folder_path,f"my-languages-{language}.so"),
    [
        os.path.join(os.getcwd(),"dependencies",f"tree-sitter-{language}")
    ]
)

# Language.build_library(
#   # Store the library in the `build` directory
#   'my-languages-java.so',

#   # Include one or more languages
#   [
#     'tree-sitter-java'
#   ]
# )

