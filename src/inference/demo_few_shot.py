from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import re

# Load v3 model
model_name = "cheralathan-m/cross-lingual-srl-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

QUESTION_MAP = {
    "ARG0": [
        "who", "किसने", "कौन", "किसको",
        "யார்", "கோனே", "காக்",
        "কোনে", "কাক", "谁",
    ],
    "ARG1": [
        "what", "क्या", "किसे",
        "என்ன", "எதை",
        "কি", "কাক", "什么",
    ],
    "ARGM-LOC": [
        "where", "कहाँ", "कहां",
        "எங்கே", "எங்கு",
        "ক'ত", "কত", "哪里", "在哪",
    ],
    "ARGM-TMP": [
        "when", "कब", "எப்போது", "কেতিয়া", "什么时候", "何时",
    ],
    "ARGM-MNR": [
        "how", "कैसे", "कैसा", "எப்படி", "কেনেকৈ", "怎么", "如何",
    ],
    "ARGM-CAU": [
        "why", "क्यों", "क्यूं", "ஏன்", "কিয়", "为什么",
    ],
}

def detect_question_type(question):
    question_lower = question.lower()
    for role, keywords in QUESTION_MAP.items():
        for keyword in keywords:
            if keyword in question_lower:
                return role
    return "ARG1"

def select_best_sentence(paragraph, question):
    sentences = re.split(r'[.।。!\?]', paragraph)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return paragraph
    q_words = set(question.lower().split())
    best_sentence = sentences[0]
    best_score = -1
    for sentence in sentences:
        s_words = set(sentence.lower().split())
        word_score = len(q_words & s_words)
        char_score = len(set(question) & set(sentence)) / max(len(set(question)), 1)
        ne_score = sum(1 for w in question.split() if w in sentence)
        score = 2 * word_score + char_score + 3 * ne_score
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence

def run_srl(sentence):
    results = pipe(sentence)
    merged = []
    for token in results:
        word = token['word'].strip()

        # FIX 1: Strip SentencePiece ▁ prefix marker (fixes Indic fragmentation)
        if word.startswith('▁'):
            word = word[1:]

        label = token['entity_group']
        score = token['score']

        if not word:
            continue

        if merged and merged[-1]['entity_group'] == label:
            prev = merged[-1]['word']
            if any(ord(c) > 0x2E80 for c in prev):    # CJK — no space
                merged[-1]['word'] += word
            elif any(ord(c) > 0x0900 for c in prev):   # Indic — no space
                merged[-1]['word'] += word
            elif word and not word[0].isalpha():        # punctuation fragment
                merged[-1]['word'] += word
            else:
                merged[-1]['word'] += ' ' + word
        else:
            merged.append({'word': word, 'entity_group': label, 'score': score})
    return merged

CONTINUOUS_MAP = {
    'playing': 'played', 'running': 'ran', 'eating': 'ate',
    'going': 'went', 'coming': 'came', 'reading': 'read',
    'writing': 'wrote', 'singing': 'sang', 'dancing': 'danced',
    'working': 'worked', 'studying': 'studied', 'teaching': 'taught',
    'helping': 'helped', 'giving': 'gave', 'taking': 'took',
    'making': 'made', 'doing': 'did', 'saying': 'said',
    'seeing': 'saw', 'looking': 'looked', 'walking': 'walked',
    'talking': 'talked', 'sitting': 'sat', 'standing': 'stood',
    'buying': 'bought', 'selling': 'sold', 'building': 'built',
    'kicking': 'kicked', 'throwing': 'threw', 'catching': 'caught',
}

def rewrite_continuous(sentence):
    words = sentence.split()
    result = []
    i = 0
    while i < len(words):
        if words[i].lower() in ('is', 'are', 'was', 'were') and i + 1 < len(words):
            next_word = words[i + 1].lower()
            if next_word in CONTINUOUS_MAP:
                result.append(CONTINUOUS_MAP[next_word])
                i += 2
                continue
            elif next_word.endswith('ing') and len(next_word) > 4:
                base = next_word[:-3]
                result.append(base + 'ed')
                i += 2
                continue
        result.append(words[i])
        i += 1
    return ' '.join(result)

def extract_answer_heuristic(sentence, target):
    words = sentence.split()
    stop_words = {'is', 'are', 'was', 'were', 'has', 'have', 'had',
                  'do', 'does', 'did', 'will', 'would', 'can', 'could',
                  'the', 'a', 'an'}
    loc_preps = {'in', 'on', 'at', 'near', 'inside', 'outside',
                 'under', 'over', 'behind', 'beside', 'into', 'to'}
    time_words = {'yesterday', 'today', 'tomorrow', 'now', 'soon',
                  'कल', 'आज', 'நேற்று', 'இன்று', 'কালি', 'আজি', '昨天', '今天'}

    verb_idx = None
    for i, w in enumerate(words):
        if w.lower() in ('is', 'are', 'was', 'were') and i + 1 < len(words):
            verb_idx = i
            break
        if w.lower() in CONTINUOUS_MAP or (w.lower().endswith('ed') and len(w) > 3):
            verb_idx = i
            break

    if target == "ARG0":
        if verb_idx is not None:
            candidates = [w for w in words[:verb_idx] if w.lower() not in stop_words]
        else:
            candidates = [words[0]] if words else []
        return ' '.join(candidates) if candidates else "Not found"

    if target == "ARG1":
        if verb_idx is not None:
            after = words[verb_idx + 1:]
            candidates = []
            for w in after:
                if w.lower() in loc_preps:
                    break
                if w.lower() not in stop_words:
                    candidates.append(w)
            return ' '.join(candidates) if candidates else "Not found"
        return "Not found"

    if target == "ARGM-LOC":
        for i, w in enumerate(words):
            if w.lower() in loc_preps and i + 1 < len(words):
                loc_words = [x for x in words[i+1:] if x.lower() not in stop_words]
                return ' '.join(loc_words) if loc_words else "Not found"
        return "Not found"

    if target == "ARGM-TMP":
        for w in words:
            if w.lower() in time_words or w in time_words:
                return w
        return "Not found"

    return "Not found"

def postprocess_answer(answer, question):
    assamese_markers = ['এ', 'ক', 'ৰ', 'লৈ', 'ত']
    tamil_particles = ['ஐ', 'இல்', 'உக்கு', 'ஆல்']
    hindi_postpositions = ['ने', 'को', 'का', 'की', 'के', 'में', 'पर', 'से']
    chinese_particles = ['了', '的', '地', '得']
    words = answer.split()
    cleaned = []
    for word in words:
        for marker in assamese_markers + tamil_particles + hindi_postpositions + chinese_particles:
            if word.endswith(marker) and len(word) > len(marker):
                word = word[:-len(marker)]
        cleaned.append(word)
    return ' '.join(cleaned).strip()

def answer_question(paragraph, question):
    sentence = select_best_sentence(paragraph, question)
    target = detect_question_type(question)

    # Attempt 1: direct SRL
    merged = run_srl(sentence)
    answer_tokens = [t['word'] for t in merged if target in t['entity_group']]

    # Attempt 2: rewrite continuous tense, retry SRL
    if not answer_tokens:
        rewritten = rewrite_continuous(sentence)
        if rewritten != sentence:
            merged2 = run_srl(rewritten)
            answer_tokens = [t['word'] for t in merged2 if target in t['entity_group']]

    # Attempt 3: heuristic positional fallback
    if not answer_tokens:
        answer = extract_answer_heuristic(sentence, target)
    else:
        if any(ord(c) > 0x2E80 for c in answer_tokens[0]):
            raw_answer = ''.join(answer_tokens)
        else:
            raw_answer = ' '.join(answer_tokens)
        answer = postprocess_answer(raw_answer, question)

    print(f"\n📝 Paragraph : {paragraph}")
    if sentence != paragraph:
        print(f"🔎 Sentence  : {sentence}")
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
        if paragraph.lower() == "quit":
            break
        answer_question(paragraph, question)
        print("-" * 60)

if __name__ == "__main__":
    print("=== CROSS-LINGUAL SRL DEMO (v3 Few-Shot) ===\n")

    para = "Ram is playing football in the park"
    answer_question(para, "Who is playing?")
    answer_question(para, "किसने खेला?")
    answer_question(para, "யார் விளையாடுகிறான்?")
    answer_question(para, "কোনে খেলিছে?")
    answer_question(para, "谁在踢球?")
    answer_question(para, "What is Ram playing?")
    answer_question(para, "क्या खेल रहा है?")
    answer_question(para, "என்ன விளையாடுகிறான்?")
    answer_question(para, "Where is Ram playing?")
    answer_question(para, "कहाँ खेल रहा है?")
    answer_question(para, "எங்கே விளையாடுகிறான்?")
    answer_question(para, "ক'ত খেলিছে?")
    answer_question(para, "哪里踢球?")

    print("\n--- Multi-sentence test ---")
    para2 = "Meena donated clothes to the flood victims. She went to the relief camp yesterday. The volunteers helped her carry the bags."
    answer_question(para2, "Who donated clothes?")
    answer_question(para2, "किसने कपड़े दान किए?")
    answer_question(para2, "When did she go?")
    answer_question(para2, "Where did she go?")

    print("\n--- Tamil paragraph test ---")
    para3 = "ராஜன் பூங்காவில் பந்தை உதைத்தான்"
    answer_question(para3, "யார் உதைத்தான்?")
    answer_question(para3, "என்ன உதைத்தான்?")
    answer_question(para3, "எங்கே உதைத்தான்?")

    print("\n--- Assamese paragraph test ---")
    para4 = "ৰামে কালি বিদ্যালয়ত গীত গালে"
    answer_question(para4, "কোনে গালে?")
    answer_question(para4, "কি গালে?")
    answer_question(para4, "ক'ত গালে?")
    answer_question(para4, "কেতিয়া গালে?")

    print("\n--- Hindi paragraph test ---")
    para5 = "राजन ने कल पार्क में गेंद को लात मारी"
    answer_question(para5, "किसने लात मारी?")
    answer_question(para5, "क्या मारी?")
    answer_question(para5, "कहाँ मारी?")
    answer_question(para5, "कब मारी?")

    print("\n--- Chinese paragraph test ---")
    para6 = "拉詹昨天在公园踢了球"
    answer_question(para6, "谁踢了球?")
    answer_question(para6, "踢了什么?")
    answer_question(para6, "哪里踢球?")

    print("\n=== NOW TRY YOUR OWN ===")
    interactive_demo()