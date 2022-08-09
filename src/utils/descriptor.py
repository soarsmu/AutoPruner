class Descriptor:
    def __init__(self, class_name, function_name, source_path, params, return_type):
        self.class_name = class_name
        self.function_name = function_name
        self.source_path = source_path
        self.params = params
        self.return_type = return_type
        self.code = ''

    def print(self):
        print("Function name: {}, params: {}, return_type: {}".format(self.function_name, self.params, self.return_type))

    def __repr__(self):
        return "Path: {}, class name: {},  Function name: {}, params: {}, return_type: {}".format(self.source_path, self.class_name, self.function_name, self.params, self.return_type)

    def __eq__(self, other):
        if not isinstance(other, Descriptor):
            return NotImplemented

        return self.class_name == other.class_name and self.function_name == other.function_name and self.source_path == other.source_path and self.params == other.params and self.return_type == other.return_type
    
    def __tocode__(self):
        output = 'class ' + self.class_name.split("/")[-1] + " { "
        output += self.return_type
        output += ' '
        output += self.function_name
        output += ' ( '
        cnt_param = 0
        for param in self.params:
            output += param
            output += ' a{}, '.format(cnt_param)
            cnt_param += 1
        output += ') '
        output += '{ return '
        output += self.return_type+"1"
        output += '; }'
        return output
    
    def __tomethod__(self):
        return self.function_name


    