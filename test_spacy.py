import spacy
import json
import csv
from datetime import datetime

# Load pretrained spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded successfully!")

# Test sentences
test_sentences = [
    "Book a flight to Paris tomorrow",
    "Cancel my order number 12345",
    "What's the weather in New York today",
    "Apply for leave from June 1st to June 5th",
    "Schedule a meeting with John at 3pm",
    "I want to check my salary",
    "Hello, how are you?",
    "Transfer $500 to my savings account",
    "Show me restaurants near Times Square",
    "What time does the store close?"
]

# Process sentences and extract entities
results = []

print("\n" + "="*60)
print("TESTING SPACY MODEL")
print("="*60 + "\n")

for sentence in test_sentences:
    doc = nlp(sentence)
    
    # Extract entities
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    
    # Extract tokens
    tokens = [token.text for token in doc]
    
    # Store result
    result = {
        "text": sentence,
        "tokens": tokens,
        "entities": entities,
        "entity_count": len(entities)
    }
    
    results.append(result)
    
    # Print output
    print(f"Text: {sentence}")
    print(f"Tokens: {tokens}")
    if entities:
        print(f"Entities found:")
        for ent in entities:
            print(f"  - {ent['text']} ({ent['label']})")
    else:
        print("  No entities found")
    print("-" * 60)

# Save to JSON
json_filename = f"spacy_test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✓ JSON output saved to: {json_filename}")

# Save to CSV
csv_filename = f"spacy_test_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Text', 'Tokens', 'Entities', 'Entity Count'])
    
    for result in results:
        entities_str = "; ".join([f"{e['text']}({e['label']})" for e in result['entities']])
        writer.writerow([
            result['text'],
            ", ".join(result['tokens']),
            entities_str if entities_str else "None",
            result['entity_count']
        ])

print(f"✓ CSV output saved to: {csv_filename}")

# Model information
print("\n" + "="*60)
print("MODEL INFORMATION")
print("="*60)
print(f"Model: {nlp.meta['name']}")
print(f"Version: {nlp.meta['version']}")
print(f"Language: {nlp.meta['lang']}")
print(f"Pipeline components: {nlp.pipe_names}")
print("\nEntity types the model can recognize:")
for label in nlp.get_pipe("ner").labels:
    print(f"  - {label}")

print("\n" + "="*60)
print("TEST COMPLETED SUCCESSFULLY!")
print("="*60)