import pandas as pd

# Zero-Shot Dataset (NER-style tagging)
zero_shot_data = {
    'text': [
        "John Smith visited New York last summer",
        "Apple Inc. announced a new product yesterday",
        "The conference will be held in London on May 15th",
        "Maria Garcia joined Microsoft as CEO in 2022",
        "The Eiffel Tower in Paris attracts millions of tourists"
    ],
    'label': [
        "B-PER I-PER O O B-LOC I-LOC O B-DATE O",
        "B-ORG I-ORG O O O O O B-DATE",
        "O O O O O O B-LOC O B-DATE I-DATE",
        "B-PER I-PER O B-ORG O O O B-DATE",
        "O B-LOC I-LOC O B-LOC O O O O"
    ]
}
zero_shot_df = pd.DataFrame(zero_shot_data)

# Code-Switch Dataset (NER-style tagging)
code_switch_data = {
    'text': [
        "I want to fly from Nairobi to Nueva York mañana",
        "Reserva un vuelo from Nairobi to London por favor",
        "Book a flight desde Nairobi a París next week",
        "Quiero un hotel en Tokyo para mi viaje",
        "Find me a restaurante in Berlin bitte"
    ],
    'label': [
        "O O O O O B-LOC O B-LOC I-LOC B-DATE",
        "O O O O B-LOC O B-LOC O O",
        "O O O O B-LOC O B-LOC B-DATE I-DATE",
        "O O O O B-LOC O O O",
        "O O O O O B-LOC O"
    ]
}
code_switch_df = pd.DataFrame(code_switch_data)

# Save datasets to CSV files
zero_shot_df.to_csv('fypenv/Datasets/Experiments/zero_shot_dataset.csv', index=False)
code_switch_df.to_csv('fypenv/Datasets/Experiments/code_switch_dataset.csv', index=False)

print("Datasets generated successfully.")
    

