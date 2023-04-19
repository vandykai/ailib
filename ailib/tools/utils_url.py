from urllib import parse

def url_parse(url, **params):
    pr = parse.urlparse(url)
    query = dict(parse.parse_qsl(pr.query))
    query.update(params)
    print(query)
    prlist = list(pr)
    prlist[4] = parse.urlencode(query)
    return parse.ParseResult(*prlist)