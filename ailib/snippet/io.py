
import os
import json

current_path = os.path.dirname(__file__)

def put_snippets(code_snippet, prefix, description, method=""):
    default_snippets_struct = {
        "prefix": "",
        "body": [],
        "description": ""
    }
    default_snippets_struct["prefix"] = prefix
    default_snippets_struct["description"] = description
    for line in code_snippet.split("\n"):
        default_snippets_struct["body"].append(line.rstrip())
    snippet = get_snippets(prefix)
    if isinstance(snippet, dict):
        if method == "append":
            snippet["body"].append("")
            snippet["body"].extend(default_snippets_struct["body"])
            with open(os.path.join(current_path, "data", prefix+".json"), "w") as f:
                f.write(json.dumps(snippet, indent=4))
        elif method == "replace":
            with open(os.path.join(current_path, "data", prefix+".json"), "w") as f:
                f.write(json.dumps(default_snippets_struct, indent=4))
        else:
            print("prefix already in snippets")
    else:
        with open(os.path.join(current_path, "data", prefix+".json"), "w") as f:
            f.write(json.dumps(default_snippets_struct, indent=4))

def get_snippets(prefix=""):
    snippets = {}
    for root, dirs, files in os.walk(os.path.join(current_path, "data")):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as f:
                    data = json.load(f)
                    snippets[file[:-5]] = data
    if prefix in snippets:
        return snippets[prefix]
    else:
        prefix_keys = []
        for key in snippets.keys():
            if key.startswith(prefix):
                prefix_keys.append(key)
        return prefix_keys