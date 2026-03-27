import unicodedata
import re
from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="cheralathan-m/cross-lingual-srl-v2",
    aggregation_strategy="simple",
)

QUESTION_MAP = {
    "ARG0": ["who","किसने","कौन","किसको","யார்","কোনে","কাক","谁","哪位"],
    "ARG1": ["what","क्या","किसे","என்ன","எதை","কি","什么","什麼"],
    "ARGM-LOC": ["where","कहाँ","कहां","எங்கே","எங்கு","ক'ত","哪里","哪裡","在哪"],
    "ARGM-TMP": ["when","कब","எப்போது","কেতিয়া","什么时候","何时"],
    "ARGM-MNR": ["how","कैसे","कैसा","எப்படி","কেনেকৈ","怎么","如何"],
    "ARGM-CAU": ["why","क्यों","क्यूं","ஏன்","কিয়","为什么","為何"],
}

STOPWORDS = {
    "who","what","where","when","how","why","did","does","is","are",
    "was","were","the","a","an","in","on","at","to","of","and",
    "or","for","with","by","from","that","this","it","do",
}

PUNCT = "?!.,;:\"\'"

_STANDALONE = {
    "a","an","the","to","of","and","or","for","with","by","from","that","this",
    "it","do","is","are","was","were","has","had","not","but","so","if","as",
    "up","out","no","he","she","we","they","his","her","its","our","who","what",
    "how","why","when","where","i","me","my","you","your","him","them","their",
    "ran","run","got","put","set","let","hit","cut","sat","ate","saw","won",
    "dog","cat","boy","car","bus","box","bed","bag","cup","sun","sky",
    "sea","air","war","law","tax","job","new","old","big","hot","red","bad",
    "few","far","low","org","inc","cook","sang",
}

_SHORT_SUFFIX = {"on","in","an","en","at"}

_FRAG_SUFFIX = re.compile(
    r"(ya|ha|ian|ant|ent|int|"
    r"ed|er|ly|al|ic|ry|ty|gy|ny|my|py|fy|"
    r"ment|ness|ity|ive|ous|ful|less|ling|nce|nse|"
    r"ble|ple|tle|dle|lete|rait|rian|phy|ess|cher|"
    r"ite|ell|ack|ick|ock|uck|age|ace|ice|ght|ion|"
    r"ight|fighter|trait|tion|room|dict|tain|ship|yani|ani|"
    r"gon|gin|gent|ount|aunt|aint|"
    r"man|gle|kram|shes|oth|ists)$",
    re.IGNORECASE
)

_ZH_LOC_STRIP_SUFFIX = {"里","内","上","中","处"}
_ZH_LOC_STRIP_PREFIX = {"在"}


def _char_script(ch):
    name = unicodedata.name(ch, "")
    if "CJK" in name:
        return "cjk"
    cat = unicodedata.category(ch)
    if cat.startswith("L") and ord(ch) > 0x0900:
        return "indic"
    return "latin"


def _detect_script(text):
    counts = {"latin": 0, "cjk": 0, "indic": 0}
    for ch in text:
        if ch.strip():
            counts[_char_script(ch)] += 1
    return max(counts, key=counts.get)


def _is_latin_token(s):
    return all(
        (unicodedata.category(c).startswith("L") and ord(c) < 0x0300) or c == " "
        for c in s if c.strip()
    )


def _is_fragment(word):
    if not word or not _is_latin_token(word):
        return False
    if word[0].isupper():
        return False
    w = word.lower()
    if w in _STANDALONE:
        return False
    if len(word) <= 2:
        return True
    if w in _SHORT_SUFFIX:
        return True
    return bool(_FRAG_SUFFIX.search(w))


def _should_merge_nonlatin(prev_word, next_word):
    if not prev_word or not next_word:
        return False
    prev_script = _char_script(prev_word[-1])
    next_script = _char_script(next_word[0])
    return prev_script == next_script and next_script in ("cjk","indic")


def _clean_zh_loc(answer):
    if not answer:
        return answer
    if answer and _char_script(answer[0]) == "cjk" and answer[0] in _ZH_LOC_STRIP_PREFIX:
        answer = answer[1:]
    if answer and _char_script(answer[-1]) == "cjk" and answer[-1] in _ZH_LOC_STRIP_SUFFIX:
        answer = answer[:-1]
    return answer


def detect_question_type(question):
    question_lower = question.lower()
    for role, keywords in QUESTION_MAP.items():
        for keyword in keywords:
            if keyword in question_lower or keyword in question:
                return role
    return "ARG1"


def select_best_sentence(paragraph, question, debug=False):
    sentences = [
        s.strip()
        for s in paragraph.replace("।",".").replace("。",".").split(".")
        if s.strip()
    ]
    if len(sentences) <= 1:
        return paragraph

    para_script = _detect_script(paragraph)
    q_script = _detect_script(question)
    cross_lingual = para_script != q_script

    question_words = {
        w.strip(PUNCT) for w in question.lower().split()
        if w.strip(PUNCT) and w.strip(PUNCT) not in STOPWORDS
    }
    q_chars = set(question) - set(" ?！？।।。.,"+PUNCT)

    best_sentence = sentences[0]
    best_score = -1

    for i, sentence in enumerate(sentences):
        sentence_words = {
            w.strip(PUNCT) for w in sentence.lower().split()
            if w.strip(PUNCT) and w.strip(PUNCT) not in STOPWORDS
        }
        word_score = len(question_words & sentence_words)
        char_score = len(q_chars & set(sentence)) / max(len(q_chars), 1)

        if cross_lingual:
            ne_score = sum(1 for qw in question.split() if qw in sentence)
            score = ne_score * 3 + word_score * 2
        else:
            score = word_score * 2 + char_score

        if debug:
            print(f"  [{i}] score={score:.3f} word={word_score} char={char_score:.3f} | {sentence[:60]}")

        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence


def merge_tokens(raw_results):
    merged = []
    for token in raw_results:
        word = token.get("word","")
        entity = token.get("entity_group") or token.get("entity","")
        score = token.get("score", 0)

        if word.startswith("##") and merged:
            if entity != merged[-1]["entity"] and score < 0.60:
                pass
            elif entity != merged[-1]["entity"] and score >= 0.60:
                merged[-1]["entity"] = entity
            merged[-1]["word"] += word[2:]
            merged[-1]["score"] = (merged[-1]["score"] + score) / 2

        elif (merged
              and entity == merged[-1]["entity"]
              and _should_merge_nonlatin(merged[-1]["word"], word)):
            merged[-1]["word"] += word
            merged[-1]["score"] = (merged[-1]["score"] + score) / 2

        elif (merged
              and entity == merged[-1]["entity"]
              and _is_latin_token(word)
              and _is_latin_token(merged[-1]["word"])
              and _is_fragment(word)):
            merged[-1]["word"] += word
            merged[-1]["score"] = (merged[-1]["score"] + score) / 2

        else:
            merged.append({"word": word, "entity": entity, "score": score})

    return merged


def answer_question(paragraph, question, debug=False):
    sentence = select_best_sentence(paragraph, question, debug=debug)
    raw_results = pipe(sentence)

    if debug:
        print("\n🔬 Raw tokens:")
        for t in raw_results:
            word = t.get("word","")
            ent = t.get("entity_group") or t.get("entity","")
            sc = t.get("score", 0)
            print(f"  {word:<20} {ent:<15} {sc:.4f}")

    tokens = merge_tokens(raw_results)

    if debug:
        print("🔗 Merged tokens:")
        for t in tokens:
            print(f"  {t['word']:<20} {t['entity']:<15} {t['score']:.4f}")

    target = detect_question_type(question)
    answer_tokens = [t["word"] for t in tokens if target in t["entity"]]

    if not answer_tokens and target == "ARG1":
        prr_tokens = [t for t in tokens if "ARGM-PRR" in t["entity"]]
        if prr_tokens and prr_tokens[0]["score"] < 0.70:
            answer_tokens = [t["word"] for t in prr_tokens]

    if answer_tokens:
        if all(_is_latin_token(t) for t in answer_tokens):
            answer = " ".join(answer_tokens)
        else:
            answer = "".join(answer_tokens)

        if answer.endswith("ই") or answer.endswith("এ"):
            answer = answer[:-1]

        CJK_CLASSIFIERS = "本些个只条张块件份"
        if (len(answer) >= 2
                and _char_script(answer[0]) == "cjk"
                and answer[0] in CJK_CLASSIFIERS):
            answer = answer[1:]

        if target == "ARGM-LOC" and answer and _char_script(answer[0]) == "cjk":
            answer = _clean_zh_loc(answer)

        if target == "ARGM-LOC" and _is_latin_token(answer):
            words = answer.split()
            if len(words) > 1 and words[-1][0].isupper():
                answer = words[-1]

    else:
        answer = "Not found"

    print(f"\n📝 Paragraph : {paragraph}")
    if sentence != paragraph:
        print(f"📌 Sentence   : {sentence}")
    print(f"❓ Question  : {question}")
    print(f"✅ Answer    : {answer}")
    print(f"🔍 Role Used : {target}")
    print()
    return answer


def interactive_demo():
    print("=" * 60)
    print("  Cross-Lingual SRL Question Answering")
    print("  Paragraph and Question can be in ANY language")
    print("=" * 60)
    print("Type 'quit' to exit\n")

    while True:
        paragraph = input("📄 Enter paragraph : ").strip()
        if paragraph.lower() == "quit":
            break
        question = input("❓ Enter question  : ").strip()
        if question.lower() == "quit":
            break
        if not question:
            print("⚠️  Please enter a question.\n")
            continue
        answer_question(paragraph, question)
        print("-" * 60)


if __name__ == "__main__":
    print("=== CROSS-LINGUAL DEMO ===\n")

    para = "The accountant prepared the quarterly financial report."
    answer_question(para, "Who prepared the report?")
    answer_question(para, "யார் அறிக்கை தயாரித்தார்?")
    answer_question(para, "किसने रिपोर्ट तैयार की?")

    para_multi = "Priya went to the market. Rahul organized a charity football match."
    answer_question(para_multi, "Who organized the match?")

    para_multi2 = "Anita baked a cake. Suresh painted the wall. Meena sang a song."
    answer_question(para_multi2, "Who painted?")
    answer_question(para_multi2, "What did Meena do?")

    print("\n=== NOW TRY YOUR OWN ===")
    interactive_demo()