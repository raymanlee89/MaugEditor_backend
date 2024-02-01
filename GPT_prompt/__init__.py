import os
from openai import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)
seed = 123

def ask_GPT(formula, prose):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        seed=seed,
        messages=[
            {"role": "system", "content": "You will find the corresponding substring in the Prose to each symbol in the Formula. All the symbol in the Formula should be in $. The substring in the Prose should be in \"."},
            {"role": "user", "content": "Formula:  $\cos \theta = \frac{<a,b>}{||a|| \cdot ||b||}$; Prose: The angle between two vectors is calculated by finding the inner product between the first and second vectors and dividing by the length of each then take the arc cosine."},
            {"role": "assistant", "content": "$\theta$: \"angle\"; $a$, $b$: \"two vectors\", \"the first and second vectors\"; $<$, $>$: \"inner product\"; $||$: \"the length\"; $\cos$: \"arc cosine\""},
            {"role": "user", "content": "Formula:  $e^{ix} = \cos(x) + i \sin(x)$; Prose: Growth in a perpendicular direction over time is circular: here are the horizontal and vertical coordinates."},
            {"role": "assistant", "content": "$e$: \"Growth\"; $i$: \"perpendicular direction\"; $x$: \"time\"; $\cos(x)$: \"horizontal\"; $i \sin(x)$: \"vertical\""},
            {"role": "user", "content": "Formula:  $\Pr(H|E) = \frac{\Pr(E|H)\Pr(H)}{\Pr(E|H)\Pr(H) + \Pr(E|\text{not } H)\Pr(\text{not } H)}$; Prose: The chance evidence is real (supports a hypothesis) is the chance of a true positive among all positives (true or false)"},
            {"role": "assistant", "content": "$\Pr$: \"chance\"; $H$: \"a hypothesis\"; $E$: \"evidence\"; $\Pr(E|H)\Pr(H)$: \"chance of a true positive\", \"true\"; $\Pr(E|\text{not } H)\Pr(\text{not } H)$: \"false\""},
            {"role": "user", "content": "Formula:  $" + formula + "$; Prose: " + prose}
        ]
    )
    return response.choices[0].message.content

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
