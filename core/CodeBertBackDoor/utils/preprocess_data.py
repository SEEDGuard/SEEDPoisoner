import json
import os

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

def preprocess_input_data(data_file,output_dir, lang):
  
    examples = []
    with open(data_file, 'r', encoding='utf-8') as pf:
        for line in pf:
            data = json.loads(line)
            doc_token = ' '.join(data['docstring_tokens'])
            code_token = ' '.join([format_str(token) for token in data['code_tokens']])
            example = (str(1), data['url'], data['func_name'], doc_token, code_token)
            examples.append('<CODESPLIT>'.join(example))
    

    dest_path = os.path.join(output_dir)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    dest_file = os.path.join(dest_path, 'raw_test_data.txt')
    print("Saving the processed input data to: " + dest_file)
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))
    return dest_file
