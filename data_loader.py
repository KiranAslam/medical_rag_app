import os
from datasets import load_dataset
from langchain_core.documents import Document


def load_medical_docs():
    docs = []

    sources = [
        ("medalpaca/medical_meadow_wikidoc", 5000),
        ("medalpaca/medical_meadow_medqa", 5000),
        ("medalpaca/medical_meadow_healthcaremagic", 5000),
        ("medalpaca/medical_meadow_mediqa", 500),
        ("keivalya/MedQuad-MedicalQnADataset", 1000),
    ]

    for dataset_name, count in sources:
        try:
            dataset = load_dataset(dataset_name, split=f"train[:{count}]")
            for row in dataset:
                input_text = row.get("input") or row.get("question") or row.get("Question", "")
                output_text = row.get("output") or row.get("answer") or row.get("Answer", "")
                if input_text and output_text:
                    content = f"Question: {input_text}\nAnswer: {output_text}"
                    docs.append(Document(page_content=content, metadata={"source": dataset_name}))
            print(f"{dataset_name}: loaded {count} docs")
        except Exception as e:
            print(f"Skipped {dataset_name}: {e}")

    print(f"Total: {len(docs)} documents")
    return docs


if __name__ == "__main__":
    docs = load_medical_docs()