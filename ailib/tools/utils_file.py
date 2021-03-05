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