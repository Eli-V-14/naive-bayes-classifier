from pathlib import Path

SMALL_DATA_PATH = Path(__file__).parent / "data" / "SmallDataset"

def load_small(filename):
    results = []

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line:
                line = line.replace("\n", "").strip()
                label, text = line[:3], line[4:]
                results.append((label, text))
            else:
                continue

    return results

if __name__ == "__main__":
    results = load_small(SMALL_DATA_PATH / "trainingSet.txt")
    print(len(results))
    print(results)