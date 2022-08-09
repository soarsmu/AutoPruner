import re
import pandas as pd
from src.utils.descriptor import Descriptor
key_to_type = {
    'B': 'byte',
    'C': 'char',
    'D': 'double',
    'F': 'float',
    'I': 'int',
    'J': 'long',
    'S': 'short',
    'Z': 'boolean',
    'V': 'void'
}

def remove_object_idx(source_path):
    prefix = source_path.split('$')[0]
    obj = source_path.split('$')[1]
    while obj.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
        obj = obj[1:]
    #
    return '$'.join([prefix, obj])

    # return source_path.split('$')[0]


def extract_function(name):
    parts = re.split('\.', name)
    function_name = parts[1]
    class_name = parts[0]
    if '$' in class_name:
        class_name = remove_object_idx(class_name)

    source_path = class_name
    if '$' in source_path:
        source_path = source_path.split('$')[0]
    
    if '$' in function_name:
        function_name = function_name.split('$')[0]

    return function_name, class_name, source_path



def extract_return_type(des):
    type = ''
    while des.startswith('['):
        type = type + '[]'
        des = des[1:]
    if des.startswith('L') and des.endswith(';'):
        type = re.split('/', des)[-1][:-1] + type
    else:
        type = key_to_type[des] + type
    if "$" in type:
        type = type.split('$')[-1]

    return type


def extract_params(des):
    if des == '':
        return []
    params = []
    parts = re.split(';', des)
    for part in parts:
        type = ''
        while len(part) > 0:
            if part.startswith('['):
                type = type + '[]'
                part = part[1:]
            else:
                if not part.startswith('L'):
                    type = key_to_type[part[0]] + type
                    params.append(type)
                    type = ''
                    part = part[1:]
                else:
                    type = re.split('/', part)[-1] + type
                    params.append(type)
                    break

    return params


def remove_params_idx(params):
    result = []
    for param in params:
        if '$' in param:
            result.append(param.split('$')[0])
        else:
            result.append(param)

    return result


def convert(descriptor):
    pattern = '\(|\)'
    descriptor = descriptor.replace(':', '')
    parts = re.split(pattern, descriptor)
    function_name, class_name, source_path = extract_function(parts[0])
    params = remove_params_idx(extract_params(parts[1]))
    return_type = extract_return_type(parts[2])

    return Descriptor(class_name, function_name, source_path, params, return_type)

if __name__ == '__main__':
    e = convert("com/bigfatplayer/hello/Calculator$calculate_args$_Fields.<init>:(Ljava/lang/String;ISLjava/lang/String;)V")
    print(e.__tocode__())