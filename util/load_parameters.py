import yaml


def load_param(path=None, filename=None, backup=False):
    try:
        if filename is not None:
            if not '.yml' in filename:
                filename = filename + '.yml'
            if path is not None:
                param_path = path + filename
                # with open(param_path, 'r') as file:
                #     global_conf = yaml.safe_load(file)
            else:
                # param_path = '../parameters/' + filename
                param_path = 'parameters/' + filename
        else:
            param_path = path
        with open(param_path, 'r') as file:
            global_conf = yaml.safe_load(file)
    except:
        raise ValueError("path or filename not found")

    if backup:
        return global_conf, param_path
    else:
        return global_conf
