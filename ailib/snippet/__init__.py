from ailib.snippet.io import get_snippets

create_var = locals()
for item_name in get_snippets():
    item = get_snippets(item_name)
    create_var[item_name] = lambda item=item:print("\n".join(item['body']))