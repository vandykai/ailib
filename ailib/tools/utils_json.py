import json
import re

def dumps_json_cpmpact(*sub, **kw):
    '''
    A wrapper of json.dumps support an optional compact_length parameter.
    compact_length take effects when indent > 0, will return a more compact
    indented result, i.e., pretty and compact json dumps.
    '''
    non_space_pattern = re.compile('[^ ]')
    compact_length = kw.pop('compact_length', None)
    r = json.dumps(*sub, **kw)
    if kw.get('indent') and compact_length:
        lines = r.split('\n')
        result_lines = [lines[0]]
        prev_space_count = None
        for line in lines[1:]:
            splitted = non_space_pattern.split(line)
            space_count = len(splitted[0])
            if space_count and prev_space_count:
                if space_count == prev_space_count\
                        or (space_count > prev_space_count and\
                            space_count - prev_space_count <= kw.get('indent')):
                    if len(line) + len(result_lines[-1]) - space_count <= compact_length\
                            and not result_lines[-1].rstrip().endswith(('],', '},')):
                        result_lines[-1] = result_lines[-1] + line[space_count:]
                        continue
            result_lines.append(line)
            prev_space_count = space_count
        r = '\n'.join(result_lines)
    return r