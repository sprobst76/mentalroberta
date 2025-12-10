"""
Deutsche Data Augmentation für MentalRoBERTa-Caps Training

Generiert mehr Trainingsdaten durch verschiedene Techniken:
1. Synonym-Ersetzung
2. Zufällige Wort-Einfügung
3. Zufällige Wort-Löschung
4. Satz-Umstellung

Verwendung:
    python augment_german.py --input german_data.json --output german_augmented.json --factor 5
"""

import json
import random
import re
import argparse
from collections import defaultdict

# Deutsche Synonyme für Mental-Health-Domain
SYNONYME = {
    # Emotionale Zustände
    "traurig": ["niedergeschlagen", "bedrückt", "bekümmert", "schwermütig", "melancholisch"],
    "glücklich": ["froh", "fröhlich", "zufrieden", "heiter", "vergnügt"],
    "wütend": ["zornig", "verärgert", "aufgebracht", "erbost", "gereizt"],
    "ängstlich": ["verängstigt", "besorgt", "furchtsam", "bange", "nervös"],
    "müde": ["erschöpft", "ausgelaugt", "kraftlos", "ermattet", "schlapp"],
    "leer": ["hohl", "ausgehöhlt", "nichtig", "taub", "gefühllos"],
    "einsam": ["allein", "isoliert", "verlassen", "abgeschnitten", "zurückgezogen"],
    
    # Mental Health Begriffe
    "deprimiert": ["niedergedrückt", "bedrückt", "down", "am Boden"],
    "gestresst": ["unter Druck", "angespannt", "überfordert", "belastet"],
    "panisch": ["in Panik", "voller Angst", "verängstigt", "entsetzt"],
    "hoffnungslos": ["aussichtslos", "verzweifelt", "resigniert", "mutlos"],
    
    # Intensitäten
    "sehr": ["extrem", "unglaublich", "wahnsinnig", "total", "richtig"],
    "immer": ["ständig", "dauernd", "permanent", "pausenlos", "fortwährend"],
    "nie": ["niemals", "zu keiner Zeit", "überhaupt nicht", "kein einziges Mal"],
    "manchmal": ["gelegentlich", "ab und zu", "hin und wieder", "zeitweise"],
    
    # Handlungen
    "weinen": ["heulen", "schluchzen", "Tränen vergießen"],
    "schlafen": ["ruhen", "dösen", "pennen", "schlummern"],
    "arbeiten": ["schaffen", "tätig sein", "werken"],
    "kämpfen": ["ringen", "streiten", "ankämpfen"],
    
    # Zeitausdrücke
    "heute": ["an diesem Tag", "momentan", "gerade"],
    "gestern": ["am Vortag", "tags zuvor"],
    "morgen": ["am nächsten Tag", "übermorgen"],
    
    # Körperliche Empfindungen
    "Schmerz": ["Leid", "Qual", "Pein", "Weh"],
    "schwer": ["belastend", "drückend", "erdrückend"],
    "eng": ["beengt", "eingeengt", "beklemmend"],
    
    # Häufige Verben
    "fühlen": ["empfinden", "spüren", "wahrnehmen"],
    "denken": ["glauben", "meinen", "annehmen"],
    "wollen": ["möchten", "wünschen", "begehren"],
    "können": ["vermögen", "in der Lage sein"],
    
    # Adjektive
    "schlimm": ["furchtbar", "schrecklich", "übel", "grauenvoll"],
    "gut": ["okay", "in Ordnung", "prima", "fein"],
    "schwierig": ["hart", "kompliziert", "mühsam", "anstrengend"],
}

# Füllwörter für Einfügung
FUELLWOERTER = [
    "wirklich", "ehrlich gesagt", "tatsächlich", "irgendwie", "halt",
    "einfach", "vielleicht", "wahrscheinlich", "definitiv", "sicherlich",
    "ich meine", "weißt du", "ich glaube", "ich denke", "quasi"
]

# Kategorie-spezifische Satzanfänge
SATZANFAENGE = {
    "depression": [
        "Ich fühle mich so", "Seit Wochen schon", "Es ist schwer zu erklären, aber",
        "Ich weiß nicht warum, aber", "Aus irgendeinem Grund", "Ich kann nicht aufhören zu",
        "Ich kämpfe gerade mit", "Es fällt mir schwer zuzugeben, dass"
    ],
    "anxiety": [
        "Ich mache mir ständig Sorgen über", "Mein Kopf hört nicht auf mit", "Ich kann nicht anders als",
        "Was ist, wenn", "Ich habe Angst, dass", "Der Gedanke an", "Ich stelle mir vor, dass",
        "Jedes Mal wenn ich daran denke"
    ],
    "bipolar": [
        "An einem Tag bin ich", "Ich wechsle zwischen", "Die Schwankungen sind", "Letzte Woche war ich",
        "Es ist wie", "Ich pendle zwischen", "Meine Stimmung ist gerade", "Manchmal fühle ich mich"
    ],
    "suicidewatch": [
        "Ich kann nicht mehr", "Ich habe darüber nachgedacht", "Ich will nicht mehr",
        "Was bringt es noch", "Ich bin so müde von", "Ich habe aufgegeben",
        "Nichts macht mehr Sinn", "Ich will nur noch"
    ],
    "offmychest": [
        "Ich muss das jemandem erzählen", "Ich habe noch nie jemandem gesagt, dass",
        "Das lastet auf mir", "Ich kann es nicht mehr für mich behalten",
        "Ich muss mich einfach auskotzen", "Niemand weiß, dass", "Ich verstecke schon lange",
        "Ich muss endlich zugeben"
    ]
}

# Zusätzliche Sätze pro Kategorie für mehr Variation
ZUSATZ_SAETZE = {
    "depression": [
        "Nichts fühlt sich mehr echt an.",
        "Die Freude ist einfach verschwunden.",
        "Ich funktioniere nur noch.",
        "Es ist wie ein grauer Schleier über allem.",
        "Ich vermisse mein altes Ich.",
        "Die Hoffnung schwindet jeden Tag mehr.",
        "Selbst atmen fühlt sich anstrengend an.",
        "Ich bin so unendlich müde."
    ],
    "anxiety": [
        "Die Angst lähmt mich komplett.",
        "Mein Herz rast ohne Grund.",
        "Ich kann nicht aufhören zu grübeln.",
        "Alles fühlt sich bedrohlich an.",
        "Die Panik kommt aus dem Nichts.",
        "Mein Körper ist in ständiger Alarmbereitschaft.",
        "Ich vermeide immer mehr.",
        "Die Sorgen hören einfach nicht auf."
    ],
    "bipolar": [
        "Die Hochs fühlen sich wie Drogen an.",
        "Dann kommt der Absturz.",
        "Stabilität kenne ich nicht.",
        "Mein Gehirn hat seinen eigenen Willen.",
        "Die Extreme sind erschöpfend.",
        "Ich weiß nie, wer ich morgen bin.",
        "Die Medikamente helfen etwas.",
        "Aber die Nebenwirkungen sind hart."
    ],
    "suicidewatch": [
        "Der Schmerz ist unerträglich.",
        "Ich will nur, dass es aufhört.",
        "Niemand würde es wirklich verstehen.",
        "Die Dunkelheit ist überwältigend.",
        "Ich halte nur noch durch.",
        "Jeden Tag weniger.",
        "Die Gedanken sind ständig da.",
        "Ich bin so müde vom Kämpfen."
    ],
    "offmychest": [
        "Endlich kann ich es aussprechen.",
        "Die Last wird leichter durch das Teilen.",
        "Ich habe so lange geschwiegen.",
        "Es fühlt sich gut an, ehrlich zu sein.",
        "Niemand in meinem Leben weiß das.",
        "Die Scham hat mich still gehalten.",
        "Aber jetzt muss es raus.",
        "Ich hoffe, jemand versteht."
    ]
}


def synonym_ersetzung(text, n=2):
    """Ersetze n Wörter durch Synonyme"""
    woerter = text.split()
    neue_woerter = woerter.copy()
    
    ersetzbar = [(i, w.lower()) for i, w in enumerate(woerter) 
                 if w.lower() in SYNONYME]
    
    if not ersetzbar:
        return text
    
    random.shuffle(ersetzbar)
    
    for i, wort in ersetzbar[:n]:
        synonyme = SYNONYME[wort]
        neue_woerter[i] = random.choice(synonyme)
    
    return ' '.join(neue_woerter)


def zufaellige_einfuegung(text, n=1):
    """Füge n zufällige Füllwörter ein"""
    woerter = text.split()
    
    for _ in range(n):
        position = random.randint(0, len(woerter))
        fuellwort = random.choice(FUELLWOERTER)
        woerter.insert(position, fuellwort)
    
    return ' '.join(woerter)


def zufaellige_loeschung(text, p=0.1):
    """Lösche Wörter mit Wahrscheinlichkeit p"""
    woerter = text.split()
    
    if len(woerter) <= 5:
        return text
    
    neue_woerter = [w for w in woerter if random.random() > p]
    
    if len(neue_woerter) < 3:
        return text
    
    return ' '.join(neue_woerter)


def satz_umstellung(text):
    """Stelle Sätze in Mehrfach-Satz-Texten um"""
    saetze = re.split(r'(?<=[.!?])\s+', text)
    
    if len(saetze) <= 1:
        return text
    
    random.shuffle(saetze)
    return ' '.join(saetze)


def satzanfang_hinzufuegen(text, label):
    """Füge kategorie-spezifischen Satzanfang hinzu"""
    if label not in SATZANFAENGE:
        return text
    
    if random.random() < 0.5:
        return text
    
    anfang = random.choice(SATZANFAENGE[label])
    
    # Erster Buchstabe klein machen wenn Anfang hinzugefügt wird
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    
    return f"{anfang} {text}"


def zusatzsatz_hinzufuegen(text, label):
    """Füge einen kategorie-spezifischen Zusatzsatz hinzu"""
    if label not in ZUSATZ_SAETZE:
        return text
    
    if random.random() < 0.6:
        return text
    
    zusatz = random.choice(ZUSATZ_SAETZE[label])
    
    # Vorne oder hinten anfügen
    if random.random() < 0.5:
        return f"{zusatz} {text}"
    else:
        return f"{text} {zusatz}"


def text_augmentieren(text, label, techniken=None):
    """Wende zufällige Augmentierungstechniken auf Text an"""
    if techniken is None:
        techniken = ['synonym', 'einfuegen', 'loeschen', 'umstellen', 'anfang', 'zusatz']
    
    # Wähle 1-3 Techniken zufällig
    anzahl = random.randint(1, 3)
    ausgewaehlt = random.sample(techniken, min(anzahl, len(techniken)))
    
    ergebnis = text
    
    for technik in ausgewaehlt:
        if technik == 'synonym':
            ergebnis = synonym_ersetzung(ergebnis, n=random.randint(1, 3))
        elif technik == 'einfuegen':
            ergebnis = zufaellige_einfuegung(ergebnis, n=random.randint(1, 2))
        elif technik == 'loeschen':
            ergebnis = zufaellige_loeschung(ergebnis, p=0.1)
        elif technik == 'umstellen':
            ergebnis = satz_umstellung(ergebnis)
        elif technik == 'anfang':
            ergebnis = satzanfang_hinzufuegen(ergebnis, label)
        elif technik == 'zusatz':
            ergebnis = zusatzsatz_hinzufuegen(ergebnis, label)
    
    return ergebnis


def datensatz_balancieren(daten):
    """Balanciere Datensatz durch Oversampling von Minderheitsklassen"""
    nach_label = defaultdict(list)
    for item in daten:
        nach_label[item['label'].lower()].append(item)
    
    max_anzahl = max(len(items) for items in nach_label.values())
    
    balanciert = []
    for label, items in nach_label.items():
        balanciert.extend(items)
        
        # Oversample wenn nötig
        while len([i for i in balanciert if i['label'].lower() == label]) < max_anzahl:
            item = random.choice(items)
            augmentiert = {
                'text': text_augmentieren(item['text'], label),
                'label': label,
                'augmentiert': True
            }
            balanciert.append(augmentiert)
    
    return balanciert


def datensatz_augmentieren(daten, faktor=3):
    """Augmentiere gesamten Datensatz"""
    augmentiert = list(daten)  # Originale behalten
    
    for item in daten:
        text = item['text']
        label = item['label'].lower()
        
        # Generiere 'faktor' augmentierte Versionen
        for _ in range(faktor):
            neuer_text = text_augmentieren(text, label)
            
            # Nur hinzufügen wenn signifikant unterschiedlich
            if neuer_text != text and len(neuer_text) > 20:
                augmentiert.append({
                    'text': neuer_text,
                    'label': label,
                    'augmentiert': True,
                    'original': text[:50] + '...'
                })
    
    return augmentiert


def main(args):
    print("=" * 60)
    print("  Deutsche Daten-Augmentierung für MentalRoBERTa-Caps")
    print("=" * 60)
    
    # Daten laden
    print(f"\n📂 Lade Daten von: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        daten = json.load(f)
    
    print(f"   Originale Samples: {len(daten)}")
    
    # Zähle nach Label
    label_counts = defaultdict(int)
    for item in daten:
        label_counts[item['label'].lower()] += 1
    
    print("\n📊 Originale Verteilung:")
    for label, count in sorted(label_counts.items()):
        print(f"   {label}: {count}")
    
    # Augmentieren
    print(f"\n🔄 Augmentiere mit Faktor {args.factor}...")
    augmentiert = datensatz_augmentieren(daten, faktor=args.factor)
    
    # Balancieren wenn gewünscht
    if args.balance:
        print("⚖️  Balanciere Klassen...")
        augmentiert = datensatz_balancieren(augmentiert)
    
    # Mischen
    random.shuffle(augmentiert)
    
    # Finale Verteilung zählen
    finale_counts = defaultdict(int)
    for item in augmentiert:
        finale_counts[item['label'].lower()] += 1
    
    print(f"\n📊 Finale Verteilung ({len(augmentiert)} gesamt):")
    for label, count in sorted(finale_counts.items()):
        print(f"   {label}: {count}")
    
    # Speichern
    print(f"\n💾 Speichere nach: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(augmentiert, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Fertig!")
    
    # Beispiele zeigen
    if args.show_examples:
        print("\n📝 Beispiel-Augmentierungen:")
        originale = [d for d in daten[:3]]
        for orig in originale:
            aug = text_augmentieren(orig['text'], orig['label'])
            print(f"\n   Original [{orig['label']}]:")
            print(f"   {orig['text'][:100]}...")
            print(f"   Augmentiert:")
            print(f"   {aug[:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deutsche Trainingsdaten augmentieren")
    parser.add_argument('--input', type=str, default='german_data.json',
                        help='Eingabe JSON-Datei')
    parser.add_argument('--output', type=str, default='german_augmented.json',
                        help='Ausgabe JSON-Datei')
    parser.add_argument('--factor', type=int, default=5,
                        help='Augmentierungsfaktor (wie viele Kopien pro Original)')
    parser.add_argument('--balance', action='store_true',
                        help='Klassen durch Oversampling balancieren')
    parser.add_argument('--show_examples', action='store_true',
                        help='Beispiel-Augmentierungen anzeigen')
    
    args = parser.parse_args()
    main(args)
