import json

def read_lines(file_path, *args):
    with open(file_path, *args) as f:
        lines = f.readlines()
    return lines

def file_de_duplication_line(file_name):
    all_line = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line not in all_line:
                all_line.append(line)
    with open(file_name, "w") as f:
        for line in all_line:
            f.write(line+"\n")

def save_to_file(json_list, file_name):
    with open(file_name, "w") as f:
        for item in json_list:
            if type(item) != str:
                item = json.dumps(item, ensure_ascii=False)
            f.write(item+'\n')