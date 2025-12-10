"""
Data Augmentation Script for MentalRoBERTa-Caps Training

Generates more training data through various augmentation techniques:
1. Synonym replacement
2. Random word insertion
3. Random word deletion
4. Sentence shuffling (for multi-sentence texts)

Usage:
    python augment_data.py --input synthetic_data.json --output augmented_data.json --factor 3
"""

import json
import random
import re
import argparse
from collections import defaultdict

# Simple synonym dictionary for mental health domain
SYNONYMS = {
    # Emotional states
    "sad": ["unhappy", "down", "low", "blue", "miserable", "gloomy"],
    "happy": ["joyful", "content", "pleased", "cheerful", "glad"],
    "angry": ["frustrated", "upset", "irritated", "mad", "annoyed"],
    "scared": ["afraid", "frightened", "terrified", "anxious", "worried"],
    "tired": ["exhausted", "drained", "fatigued", "worn out", "spent"],
    "empty": ["hollow", "void", "numb", "blank", "vacant"],
    "alone": ["lonely", "isolated", "solitary", "abandoned", "disconnected"],
    
    # Mental health terms
    "depressed": ["down", "low", "dejected", "despondent"],
    "anxious": ["worried", "nervous", "uneasy", "restless", "on edge"],
    "panic": ["terror", "dread", "alarm", "fear"],
    "worry": ["concern", "stress", "anxiety", "unease"],
    "stress": ["pressure", "tension", "strain", "burden"],
    
    # Actions
    "cry": ["weep", "sob", "tear up"],
    "sleep": ["rest", "slumber", "nap"],
    "eat": ["consume", "have food", "dine"],
    "work": ["job", "employment", "career", "profession"],
    "help": ["support", "assist", "aid"],
    
    # Intensifiers
    "very": ["really", "extremely", "incredibly", "so", "truly"],
    "always": ["constantly", "continuously", "perpetually", "forever"],
    "never": ["not once", "at no time", "not ever"],
    "sometimes": ["occasionally", "at times", "now and then", "periodically"],
    
    # Time expressions
    "today": ["this day", "right now", "currently"],
    "yesterday": ["the day before", "the previous day"],
    "tomorrow": ["the next day", "the following day"],
    "week": ["seven days", "past week", "last week"],
    "month": ["past month", "last month", "30 days"],
    
    # Physical sensations
    "pain": ["hurt", "ache", "discomfort", "suffering"],
    "heavy": ["weighted", "burdened", "loaded"],
    "tight": ["tense", "constricted", "strained"],
    "racing": ["pounding", "rapid", "fast", "quick"],
    
    # Common verbs
    "feel": ["sense", "experience", "perceive"],
    "think": ["believe", "consider", "suppose"],
    "want": ["wish", "desire", "need", "crave"],
    "know": ["understand", "realize", "recognize"],
    "can't": ["cannot", "unable to", "can not"],
}

# Filler words that can be inserted
FILLER_WORDS = [
    "really", "honestly", "actually", "basically", "literally",
    "just", "maybe", "probably", "definitely", "certainly",
    "I mean", "you know", "I think", "I guess", "I feel like"
]

# Sentence starters for variation
SENTENCE_STARTERS = {
    "depression": [
        "I've been feeling", "Lately I've noticed", "It's been hard because",
        "I don't know why but", "For some reason", "I can't explain it but",
        "I've been struggling with", "It's difficult to admit but"
    ],
    "anxiety": [
        "I keep worrying about", "My mind won't stop", "I can't help but think",
        "What if", "I'm terrified that", "The thought of", "I keep imagining",
        "Every time I think about"
    ],
    "bipolar": [
        "One day I'm", "I went from", "The swings are", "Last week I was",
        "It's like", "I cycle between", "My mood just", "Sometimes I feel"
    ],
    "suicidewatch": [
        "I can't take it anymore", "I've been thinking about", "I don't want to",
        "What's the point of", "I'm so tired of", "I've given up on",
        "Nothing matters", "I just want"
    ],
    "offmychest": [
        "I need to tell someone", "I've never told anyone this but",
        "This has been weighing on me", "I can't keep this inside anymore",
        "I just need to vent", "Nobody knows this but", "I've been hiding",
        "I finally have to admit"
    ]
}


def synonym_replacement(text, n=2):
    """Replace n words with synonyms"""
    words = text.split()
    new_words = words.copy()
    
    replaceable = [(i, w.lower()) for i, w in enumerate(words) 
                   if w.lower() in SYNONYMS]
    
    if not replaceable:
        return text
    
    random.shuffle(replaceable)
    
    for i, word in replaceable[:n]:
        synonyms = SYNONYMS[word]
        new_words[i] = random.choice(synonyms)
    
    return ' '.join(new_words)


def random_insertion(text, n=1):
    """Insert n random filler words"""
    words = text.split()
    
    for _ in range(n):
        insert_pos = random.randint(0, len(words))
        filler = random.choice(FILLER_WORDS)
        words.insert(insert_pos, filler)
    
    return ' '.join(words)


def random_deletion(text, p=0.1):
    """Randomly delete words with probability p"""
    words = text.split()
    
    if len(words) <= 5:
        return text
    
    new_words = [w for w in words if random.random() > p]
    
    if len(new_words) < 3:
        return text
    
    return ' '.join(new_words)


def sentence_shuffle(text):
    """Shuffle sentences in multi-sentence text"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 1:
        return text
    
    random.shuffle(sentences)
    return ' '.join(sentences)


def add_starter(text, label):
    """Add a sentence starter appropriate for the label"""
    if label not in SENTENCE_STARTERS:
        return text
    
    if random.random() < 0.5:
        return text
    
    starter = random.choice(SENTENCE_STARTERS[label])
    
    # Make first letter lowercase if adding starter
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    
    return f"{starter} {text}"


def augment_text(text, label, techniques=None):
    """Apply random augmentation techniques to text"""
    if techniques is None:
        techniques = ['synonym', 'insert', 'delete', 'shuffle', 'starter']
    
    # Choose 1-2 techniques randomly
    num_techniques = random.randint(1, 2)
    chosen = random.sample(techniques, min(num_techniques, len(techniques)))
    
    result = text
    
    for technique in chosen:
        if technique == 'synonym':
            result = synonym_replacement(result, n=random.randint(1, 3))
        elif technique == 'insert':
            result = random_insertion(result, n=random.randint(1, 2))
        elif technique == 'delete':
            result = random_deletion(result, p=0.1)
        elif technique == 'shuffle':
            result = sentence_shuffle(result)
        elif technique == 'starter':
            result = add_starter(result, label)
    
    return result


def balance_dataset(data):
    """Balance dataset by oversampling minority classes"""
    by_label = defaultdict(list)
    for item in data:
        by_label[item['label'].lower()].append(item)
    
    max_count = max(len(items) for items in by_label.values())
    
    balanced = []
    for label, items in by_label.items():
        balanced.extend(items)
        
        # Oversample if needed
        while len([i for i in balanced if i['label'].lower() == label]) < max_count:
            item = random.choice(items)
            augmented = {
                'text': augment_text(item['text'], label),
                'label': label,
                'augmented': True
            }
            balanced.append(augmented)
    
    return balanced


def augment_dataset(data, factor=2):
    """Augment entire dataset"""
    augmented = list(data)  # Keep originals
    
    for item in data:
        text = item['text']
        label = item['label'].lower()
        
        # Generate 'factor' augmented versions
        for _ in range(factor):
            new_text = augment_text(text, label)
            
            # Only add if significantly different
            if new_text != text and len(new_text) > 20:
                augmented.append({
                    'text': new_text,
                    'label': label,
                    'augmented': True,
                    'original': text[:50] + '...'
                })
    
    return augmented


def main(args):
    print("=" * 60)
    print("  Data Augmentation for MentalRoBERTa-Caps")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Original samples: {len(data)}")
    
    # Count by label
    label_counts = defaultdict(int)
    for item in data:
        label_counts[item['label'].lower()] += 1
    
    print("\n📊 Original distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")
    
    # Augment
    print(f"\n🔄 Augmenting with factor {args.factor}...")
    augmented = augment_dataset(data, factor=args.factor)
    
    # Balance if requested
    if args.balance:
        print("⚖️  Balancing classes...")
        augmented = balance_dataset(augmented)
    
    # Shuffle
    random.shuffle(augmented)
    
    # Count final distribution
    final_counts = defaultdict(int)
    for item in augmented:
        final_counts[item['label'].lower()] += 1
    
    print(f"\n📊 Final distribution ({len(augmented)} total):")
    for label, count in sorted(final_counts.items()):
        print(f"   {label}: {count}")
    
    # Save
    print(f"\n💾 Saving to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Done!")
    
    # Show some examples
    if args.show_examples:
        print("\n📝 Example augmentations:")
        originals = [d for d in data[:3]]
        for orig in originals:
            aug = augment_text(orig['text'], orig['label'])
            print(f"\n   Original: {orig['text'][:80]}...")
            print(f"   Augmented: {aug[:80]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment training data")
    parser.add_argument('--input', type=str, default='synthetic_data.json',
                        help='Input JSON file')
    parser.add_argument('--output', type=str, default='augmented_data.json',
                        help='Output JSON file')
    parser.add_argument('--factor', type=int, default=3,
                        help='Augmentation factor (how many copies per original)')
    parser.add_argument('--balance', action='store_true',
                        help='Balance classes by oversampling')
    parser.add_argument('--show_examples', action='store_true',
                        help='Show example augmentations')
    
    args = parser.parse_args()
    main(args)
