import os
from openai import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
print("OPENAI_API_KEY: ", OPENAI_API_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)
seed = 123
model = "gpt-4-0125-preview"

LinkPrompt = ""
with open('GPT_prompt/LinkPrompt.txt', 'r') as f:
    LinkPrompt = f.read()

Def2SymPrompt = ""
with open('GPT_prompt/Def2SymPrompt.txt', 'r') as f:
    Def2SymPrompt = f.read()

Sym2DefPrompt = ""
with open('GPT_prompt/Sym2DefPrompt.txt', 'r') as f:
    Sym2DefPrompt = f.read()

def initial_prompt(target, initial, shots):
    messages = []
    messages.append({
        "role": "system",
        "content": initial
    })
    for shot in shots:
        messages.append({"role": "user", "content": shot["Q"]})
        messages.append({"role": "assistant", "content": shot["A"]})
    messages.append({"role": "user", "content": target})
    return messages

def add_conversation(messages, conversation):
    for i in range(len(conversation)):
        role = "assistant"
        if(i % 2 == 1):
            role = "user"
        messages.append({"role": role, "content": conversation[i]})
    return messages

def ask_GPT(messages):
    GPT_response = client.chat.completions.create(
        model=model,
        seed=seed,
        temperature=0,
        messages=messages
    )
    return GPT_response.choices[0].message.content

def ask_GPT_link(formula, prose, conversation):
    shots = [
        {
            "Q": "Formula: $\\Pr(H|E) = \\frac{\\Pr(E|H)\\Pr(H)}{\\Pr(E|H)\\Pr(H) + \\Pr(E|\\text{not } H)\\Pr(\\text{not } H)}$; Prose: \"The chance evidence is real (supports a hypothesis) is the chance of a true positive among all positives (true or false)\"",
            "A": "$\\Pr(H|E)$: \"The chance evidence is real\"; $\\Pr(E|H)\\Pr(H)$: \"the chance of a true positive\", \"true\"; $\\Pr(E|\\text{not } H)\\Pr(\\text{not } H)$: \"false\""
        },
        {
            "Q": "Formula: $\\cos \\theta = \\frac{<a,b>}{||a|| \cdot ||b||}$; Prose: \"The angle between two vectors is calculated by finding the inner product between the first and second vectors and dividing by the length of each then take the arc cosine.\"",
            "A": "$\\theta$: \"angle\"; $a$, $b$: \"two vectors\", \"the first and second vectors\"; $<$, $>$: \"inner product\"; $||$: \"the length\"; $\\cos$: \"arc cosine\""
        },
        {
            "Q": "Formula: $e^{ix} = \\cos(x) + i \\sin(x)$; Prose: \"Growth in a perpendicular direction over time is circular: here are the horizontal and vertical coordinates.\"",
            "A": "$e$: \"Growth\"; $i$: \"perpendicular direction\"; $x$: \"time\"; $\\cos(x)$: \"horizontal\"; $i \sin(x)$: \"vertical\""
        },
        {
            "Q": "Formula: $r_s = \\frac{2GM}{c^{2}}$; Prose: \"The Schwarzschild radius is given as, where $G$ is the gravitational constant, $M$ is the object mass, and $c$ is the speed of light.\"",
            "A": "$r_s$: \"Schwarzschild radius\"; $G$: \"$G$\", \"the gravitational constant\"; $M$: \"$M$\", \"the object mass\"; $c$: \"$c$\", \"the speed of light\""
        }
    ]
    target = "Formula: $" + formula + "$; Prose: \"" + prose + "\""
    messages = initial_prompt(target, LinkPrompt, shots)
    messages = add_conversation(messages, conversation)
    response = ask_GPT(messages)
    return response

def ask_GPT_definition(formula, prose, symbols, conversation):
    symbolsString = ""
    for i in range(len(symbols)):
        if i == 0:
            symbolsString = symbolsString + "$" + symbols[i] + "$"
        else:
            symbolsString = symbolsString + ", $" + symbols[i] + "$"
    
    shots = [
        {
            "Q": "Formula: $\\Pr(H|E) = \\frac{\\Pr(E|H)\\Pr(H)}{\\Pr(E|H)\\Pr(H) + \\Pr(E|\\text{not } H)\\Pr(\\text{not } H)}$; Prose: \"The chance evidence is real (supports a hypothesis) is the chance of a true positive among all positives (true or false)\"; Symbols: $\\Pr(E|H)$",
            "A": "$\\Pr(E|H)$: \"The chance evidence is real\""
        },
        {
            "Q": "Formula: $\\cos \\theta = \\frac{<a,b>}{||a|| \\cdot ||b||}$; Prose: \"The angle between two vectors is calculated by finding the inner product between the first and second vectors and dividing by the length of each then take the arc cosine.\"; Symbols: $a$, $b$",
            "A": "$a$, $b$: \"two vectors\", \"the first and second vectors\""
        },
        {
            "Q": "Formula: $e^{ix} = \\cos(x) + i \\sin(x)$; Prose: \"Growth in a perpendicular direction over time is circular: here are the horizontal and vertical coordinates.\"; Symbols: $e$",
            "A": "$e$: \"Growth\""
        },
        {
            "Q": "Formula: $r_s = \\frac{2GM}{c^{2}}$; Prose: \"The Schwarzschild radius is given as, where $G$ is the gravitational constant, $M$ is the object mass, and $c$ is the speed of light.\"; Symbols: $G$",
            "A": "$G$: \"$G$\", \"the gravitational constant\""
        }
    ]
    target = "Formula: $" + formula + "$; Prose: \"" + prose + "\"; Symbols: $" + symbolsString
    messages = initial_prompt(target, Sym2DefPrompt, shots)
    messages = add_conversation(messages, conversation)
    response = ask_GPT(messages)
    return response

def ask_GPT_symbol(formula, prose, definitions, conversation):
    definitionsString = ""
    for i in range(len(definitions)):
        if i == 0:
            definitionsString = definitionsString + "\"" + definitions[i] + "\""
        else:
            definitionsString = definitionsString + ", \"" + definitions[i] + "\""
    
    shots = [
        {
            "Q": "Formula: $\\Pr(H|E) = \\frac{\\Pr(E|H)\\Pr(H)}{\\Pr(E|H)\\Pr(H) + \\Pr(E|\\text{not } H)\\Pr(\\text{not } H)}$; Prose: \"The chance evidence is real (supports a hypothesis) is the chance of a true positive among all positives (true or false)\"; Definitions: \"the chance of a true positive\", \"true\"",
            "A": "$\\Pr(E|H)\\Pr(H)$: \"the chance of a true positive\", \"true\""    
        },
        {
            "Q": "Formula: $\\cos \\theta = \\frac{<a,b>}{||a|| \cdot ||b||}$; Prose: \"The angle between two vectors is calculated by finding the inner product between the first and second vectors and dividing by the length of each then take the arc cosine.\"; Definitions: \"two vectors\", \"the first and second vectors\"",
            "A": "$a$, $b$: \"two vectors\", \"the first and second vectors\""
        },
        {
            "Q": "Formula: $\\cos \\theta = \\frac{<a,b>}{||a|| \cdot ||b||}$; Prose: \"The angle between two vectors is calculated by finding the inner product between the first and second vectors and dividing by the length of each then take the arc cosine.\"; Definitions: \"inner product\"",
            "A": "$<$, $>$: \"inner product\""
        },
        {
            "Q": "Formula: $r_s = \\frac{2GM}{c^{2}}$; Prose: \"The Schwarzschild radius is given as, where $G$ is the gravitational constant, $M$ is the object mass, and $c$ is the speed of light.\"; Definitions: \"the gravitational constant\"",
            "A": "$G$: \"the gravitational constant\""
        }
    ]
    target = "Formula: $" + formula + "$; Prose: \"" + prose + "\"; Definitions: $" + definitionsString
    messages = initial_prompt(target, LinkPrompt, shots)
    messages = add_conversation(messages, conversation)
    response = ask_GPT(messages)
    return response

def parse_response(response):
    links = response.split("; ")
    result = []
    for link in links:
        split = link.find(": ")
        symbols = [s.replace("$", "") for s in link[:split].split(", ")]
        terms = [t.replace('"', '') for t in link[split+2:].split(", ")]
        item = {"symbols": symbols, "terms": terms}
        result.append(item)
    return result

def check_format(response):
    first_split = response.find(": ")
    if first_split == -1:
        return False
    return True