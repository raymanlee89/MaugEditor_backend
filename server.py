# for  Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
# for LinksExtraction
from LinksExtraction import device, NER_tokenizer, NER_model, RE_tokenizer, RE_model, NER, RE, post_processing
from GPT_prompt import ask_GPT_link, parse_link_response, ask_GPT_definition, parse_definition_response, ask_GPT_symbol, parse_symbol_response

def parse_request(request, type = "default"):
    string = request.get_data().decode("utf-8")
    split = string.find("<;>")
    formula = string[:split]
    prose = string[split+3:]
    if type == "with item":
        nextSplit = string.find("<;>", split+3)
        prose = string[split+3:nextSplit]
        items = string[nextSplit+3:].split("<,>")
        print("items:", items)
        return formula, prose, items
    return formula, prose

def get_item_location(item, string):
    result = []
    start = string.find(item)
    while(start != -1):
        end = start + len(item)
        loc = {"text": item, "start": start, "end": end}
        result.append(loc)
        start = string.find(item, end + 1)
    return result

app = Flask(__name__)
CORS(app)

@app.route("/links", methods=["POST"])
def create_links():
    print("==========================NER & RE==========================")
    print("==========================New POST==========================")
    formula, prose = parse_request(request)
    print("formula\n", formula)
    print("\nprose\n", prose)
    entities = NER(prose, NER_tokenizer, NER_model, device)
    print("\nentities\n", entities)
    relations = RE(prose, entities, RE_tokenizer, RE_model, device)
    print("\nrelations\n", relations)
    links_in_prose = post_processing(entities, relations)
    print("\nlinks_in_prose\n", links_in_prose)
    links = []
    for link in links_in_prose:
        term_locs = []
        # make sure that the symbol text is unique
        target_texts = set()
        for item in link:
            if item["label"] == "SYMBOL":
                target_texts.add(item["text"])
            term_locs.append({"text": item["text"], "start": item["start"], "end": item["end"]})
        symbol_locs = []
        for text in target_texts:
            locs = get_item_location(text, formula)
            symbol_locs = symbol_locs + locs
        links.append({"symbols": symbol_locs, "terms": term_locs})
    print("\nlinks\n", links)
    print("============================================================")
    return jsonify(links)

@app.route("/links_GPT", methods=["POST"])
def create_links_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    formula, prose = parse_request(request)
    print("formula\n", formula)
    print("\nprose\n", prose)
    res = ask_GPT_link(formula, prose)
    print("\nres\n", res)
    links_text = parse_link_response(res)
    print("\nlinks_text\n", links_text)
    links = []
    for link in links_text:
        symbol_locs = []
        term_locs = []
        for symbol in link["symbols"]:
            locs = get_item_location(symbol, formula)
            symbol_locs = symbol_locs + locs
        for term in link["terms"]:
            locs = get_item_location(term, prose)
            term_locs = term_locs + locs
        links.append({"symbols": symbol_locs, "terms": term_locs})
    print("\nlinks\n", links)
    print("============================================================")
    return jsonify(links)

@app.route("/definition_GPT", methods=["POST"])
def create_definition_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    formula, prose, symbols = parse_request(request, "with item")
    print("formula\n", formula)
    print("\nprose\n", prose)
    print("\nsymbols\n", symbols)
    res = ask_GPT_definition(formula, prose, symbols)
    print("\nres\n", res)
    definitions_text = parse_definition_response(res)
    print("\ndefinitions_text\n", definitions_text)
    definitions = []
    for term in definitions_text:
        locs = get_item_location(term, prose)
        definitions = definitions + locs
    print("\ndefinitions\n", definitions)
    print("============================================================")
    return jsonify(definitions)

@app.route("/symbol_GPT", methods=["POST"])
def create_symbol_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    formula, prose, definitions = parse_request(request, "with item")
    print("formula\n", formula)
    print("\nprose\n", prose)
    print("\ndefinitions\n", definitions)
    res = ask_GPT_symbol(formula, prose, definitions)
    print("\nres\n", res)
    symbols_text = parse_symbol_response(res)
    print("\nsymbols_text\n", symbols_text)
    symbols = []
    for symbol in symbols_text:
        locs = get_item_location(symbol, formula)
        symbols = symbols + locs
    print("\nsymbols\n", symbols)
    print("============================================================")
    return jsonify(symbols)

if __name__ == "__main__":
    app.run(port=5000, debug=True)