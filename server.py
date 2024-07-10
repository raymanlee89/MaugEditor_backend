# for  Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
# for LinksExtraction
from LinksExtraction import device, NER_tokenizer, NER_model, RE_tokenizer, RE_model, NER, RE, post_processing
from GPT_prompt import ask_GPT_link, ask_GPT_definition, ask_GPT_symbol, parse_response, check_format

def get_item_location(item, string):
    result = []
    start = string.find(item)
    while(start != -1):
        end = start + len(item)
        loc = {"text": item, "start": start, "end": end}
        result.append(loc)
        start = string.find(item, end + 1)
    return result

def parse_links_text(links_text, formula, prose):
    links = []
    for link in links_text:
        symbol_locs = []
        term_locs = []
        for symbol in link["symbols"]:
            locs = get_item_location(symbol, formula)
            symbol_locs = symbol_locs + locs
            # Symbols in prose (formatted case)
            locs = get_item_location("$" + symbol + "$", prose)
            if len(locs) > 0:
                term_locs = term_locs + locs
        for term in link["terms"]:
            locs = get_item_location(term, prose)
            term_locs = term_locs + locs
        links.append({"symbols": symbol_locs, "terms": term_locs})
    return links

app = Flask(__name__)
CORS(app)

@app.route("/links", methods=["POST"])
def create_links():
    print("==========================NER & RE==========================")
    print("==========================New POST==========================")
    content = request.json
    formula = content["formula"]
    prose = content["prose"]
    print("formula\n", formula)
    print("\nprose\n", prose)
    entities = NER(prose, NER_tokenizer, NER_model, device)
    print("\nentities\n", entities)
    relations = RE(prose, entities, RE_tokenizer, RE_model, device)
    print("\nrelations\n", relations)
    links_in_prose = post_processing(entities, relations)
    print("\nlinks_in_prose\n", links_in_prose)
    links = []
    rawString = ""
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
        # create GPT style raw string
        symbol_string = ""
        term_string = ""
        for symbol in symbol_locs:
            symbol_string += "$" + symbol["text"] + "$, "
        if(len(symbol_string) > 0):
            symbol_string = symbol_string[:-2]
        for term in term_locs:
            term_string += "\"" + term["text"] + "\", "
        if(len(term_string) > 0):
            term_string = term_string[:-2]
        rawString += symbol_string + ": " + term_string + "; "
    if(len(rawString) > 0):
        rawString = rawString[:-2]
    print("\nlinks\n", links)
    print("============================================================")
    return jsonify({"links": links, "rawString": rawString})

@app.route("/links_GPT", methods=["POST"])
def create_links_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    content = request.json
    formula = content["formula"]
    prose = content["prose"]
    conversation = content["conversation"]
    print("formula\n", formula)
    print("\nprose\n", prose)
    print("\nconversation\n", conversation)
    res = ask_GPT_link(formula, prose, conversation)
    print("\nres\n", res)
    if not check_format(res):
        return jsonify({"warning": "Inappropriate feedback", "rawString": res})
    links_text = parse_response(res)
    print("\nlinks_text\n", links_text)
    links = parse_links_text(links_text, formula, prose)
    print("\nlinks\n", links)
    print("============================================================")
    return jsonify({"links": links, "rawString": res})

@app.route("/definition_GPT", methods=["POST"])
def create_definition_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    content = request.json
    formula = content["formula"]
    prose = content["prose"]
    symbols = content["symbol"]
    conversation = content["conversation"]
    print("formula\n", formula)
    print("\nprose\n", prose)
    print("\nsymbols\n", symbols)
    print("\nconversation\n", conversation)
    res = ask_GPT_definition(formula, prose, symbols, conversation)
    print("\nres\n", res)
    if not check_format(res):
        return jsonify({"warning": "Inappropriate feedback", "rawString": res})
    links_text = parse_response(res)
    print("\nlinks_text\n", links_text)
    links = parse_links_text(links_text, formula, prose)
    print("\nlinks\n", links)
    print("============================================================")
    return jsonify({"link": links[0], "rawString": res})

@app.route("/symbol_GPT", methods=["POST"])
def create_symbol_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    content = request.json
    formula = content["formula"]
    prose = content["prose"]
    definitions = content["definition"]
    conversation = content["conversation"]
    print("formula\n", formula)
    print("\nprose\n", prose)
    print("\ndefinitions\n", definitions)
    print("\nconversation\n", conversation)
    res = ask_GPT_symbol(formula, prose, definitions, conversation)
    print("\nres\n", res)
    if not check_format(res):
        return jsonify({"warning": "Inappropriate feedback", "rawString": res})
    links_text = parse_response(res)
    print("\nlinks_text\n", links_text)
    links = parse_links_text(links_text, formula, prose)
    print("\nlinks\n", links)
    print("============================================================")
    return jsonify({"link": links[0], "rawString": res})

if __name__ == "__main__":
    app.run(port=5000, debug=True)