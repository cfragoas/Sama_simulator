import yaml


def load_param(path=None, filename=None):
    try:
        if filename is not None:
            if not '.yml' in filename:
                filename = filename + '.yml'
            if path is not None:
                with open(path + filename, 'r') as file:
                    global_conf = yaml.safe_load(file)
            else:
                with open('../parameters/' + filename, 'r') as file:
                    global_conf = yaml.safe_load(file)
        else:
            with open(path, 'r') as file:
                global_conf = yaml.safe_load(file)
    except:
        raise ValueError("path or filename not found")

    return global_conf
