from transformers import pipeline

pipe = pipeline(
    "token-classification",
    model="MRC005/cross-lingual-srl"
)

# Comprehensive question word mapping for all languages
QUESTION_MAP = {
    # WHO → ARG0 (subject/agent)
    "ARG0": [
        "who",                          # English
        "किसने", "कौन", "किसको",         # Hindi
        "யார்",                          # Tamil
        "কোনে", "কাক",                   # Assamese
    ],
    # WHAT → ARG1 (object)
    "ARG1": [
        "what",                         # English
        "क्या", "किसे", "क्या को",       # Hindi
        "என்ன", "எதை",                   # Tamil
        "কি", "কাক",                     # Assamese
    ],
    # WHERE → location
    "ARGM-LOC": [
        "where",                        # English
        "कहाँ", "कहां",                  # Hindi
        "எங்கே", "எங்கு",                # Tamil
        "ক'ত", "কত",                     # Assamese
    ],
    # WHEN → time
    "ARGM-TMP": [
        "when",                         # English
        "कब",                           # Hindi
        "எப்போது",                       # Tamil
        "কেতিয়া",                        # Assamese
    ],
    # HOW → manner
    "ARGM-MNR": [
        "how",                          # English
        "कैसे", "कैसा",                  # Hindi
        "எப்படி",                        # Tamil
        "কেনেকৈ",                        # Assamese
    ],
    # WHY → cause
    "ARGM-CAU": [
        "why",                          # English
        "क्यों", "क्यूं",               # Hindi
        "ஏன்",                           # Tamil
        "কিয়",                           # Assamese
    ],
}

def detect_question_type(question):
    """Detect which semantic role the question is asking about."""
    question_lower = question.lower()
    for role, keywords in QUESTION_MAP.items():
        for keyword in keywords:
            if keyword in question_lower:
                return role
    return "ARG1"  # default fallback

def answer_question(paragraph, question):
    """Answer any question about any paragraph in any language."""
    
    # Tag the paragraph
    results = pipe(paragraph)
    
    # Merge subword tokens
    merged = []
    for token in results:
        word = token['word']
        label = token['entity']
        if word.startswith("##") and merged:
            merged[-1]['word'] += word[2:]
        else:
            merged.append({'word': word, 'entity': label})
    
    # Detect what role the question is asking for
    target = detect_question_type(question)
    
    # Find matching tokens
    answer_tokens = [
        t['word'] for t in merged
        if target in t['entity']
    ]
    
    answer = " ".join(answer_tokens) if answer_tokens else "Not found"
    
    print(f"\n📝 Paragraph : {paragraph}")
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
        
        answer_question(paragraph, question)
        print("-" * 60)

if __name__ == "__main__":
    # Test cross-lingual combinations
    print("=== CROSS-LINGUAL DEMO ===\n")

    # English para, questions in all languages
    para = "Ram is playing football in the park"
    answer_question(para, "Who is playing?")           # English
    answer_question(para, "किसने खेला?")               # Hindi question
    answer_question(para, "யார் விளையாடுகிறான்?")      # Tamil question
    answer_question(para, "কোনে খেলিছে?")              # Assamese question
    answer_question(para, "What is Ram playing?")      # English
    answer_question(para, "क्या खेल रहा है?")          # Hindi question
    
    # Then interactive mode
    print("\n=== NOW TRY YOUR OWN ===")
    interactive_demo()
