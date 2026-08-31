from datasets import load_dataset
from itertools import islice

def sample_personas(n=10, subset="elite_persona", seed=42, shuffle_buffer=1_000):
    """
    Sample n personas from PersonaHub without downloading the full dataset.

    subset options include (check the dataset repo for the current list):
    'persona', 'elite_persona', 'instruction', 'knowledge',
    'math', 'npc', 'reasoning', 'tool', ...
    """
    ds = load_dataset(
        "proj-persona/PersonaHub",
        subset,
        split="train",
        streaming=True,   # <-- key part: no full download, reads records lazily
    ).select_columns(["persona"])

    # Shuffle with a reservoir buffer (approximate shuffling for streaming datasets)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    # Pull just n examples
    samples = list(islice(ds, n))
    return samples