"""
Fetches arXiv paper abstracts to build our RAG corpus.

Acquisition stage of the pipeline. 
Everything downstream(chunking, indexing, retrieval) depends on this data 
being clean and well-structured, so we save it as simple JSON records:
{id, title, abstract, authors, published, url}
"""

import arxiv
import json
from pathlib import Path

# Focused category = coherent vocabulary = meaningful retrieval challenge.
# cs.CL  = Computation and Language(NLP)
# cs.AI  = Artificial Intelligence
CATEGORY = "cat:cs.CL"
MAX_RESULTS = 200 

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "arxiv_corpus.json"


def fetch_corpus(query: str = CATEGORY, max_results: int = MAX_RESULTS) -> list[dict]:
    #Queries the arXiv API and returns a list of paper records.
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # recent papers = current terminology
    )

    records = []
    for result in client.results(search):
        records.append({
            "id": result.entry_id.split("/")[-1],   # e.g. "2409.12345v1"
            "title": result.title.strip().replace("\n", " "),
            "abstract": result.summary.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "published": result.published.isoformat(),
            "url": result.entry_id,
        })
    return records


def save_corpus(records: list[dict], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} papers to {path}")


if __name__ == "__main__":
    papers = fetch_corpus()
    save_corpus(papers)