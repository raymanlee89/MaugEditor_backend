# for  Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
# for LinksExtraction
from LinksExtraction import device, NER_tokenizer, NER_model, RE_tokenizer, RE_model, NER, RE, post_processing
from GPT_prompt import ask_GPT, parse_response

def parse_request(request):
    string = request.get_data().decode("utf-8")
    split = string.find("<;>")
    formula = string[:split]
    prose = string[split+3:]
    print("formula:", formula)
    print("prose:", prose)
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
    entities = NER(prose, NER_tokenizer, NER_model, device)
    print("\n\nentities\n", entities)
    relations = RE(prose, entities, RE_tokenizer, RE_model, device)
    print("\n\nrelations\n", relations)
    links_in_prose = post_processing(entities, relations)
    print("\n\nlinks_in_prose\n", links_in_prose)
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
    print("\n\nlinks\n", links)
    print("============================================================")
    return jsonify(links)

@app.route("/links_GPT", methods=["POST"])
def create_links_GPT():
    print("============================GPT=============================")
    print("==========================New POST==========================")
    formula, prose = parse_request(request)
    res = ask_GPT(formula, prose)
    print("\n\nres\n", res)
    links_text = parse_response(res)
    print("\n\nlinks_text\n", links_text)
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
    print("\n\nlinks\n", links)
    print("============================================================")
    return jsonify(links)

if __name__ == "__main__":
    app.run(port=5000, debug=True)