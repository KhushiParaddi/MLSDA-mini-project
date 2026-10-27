import os
import re
import json

def parse_markdown_to_chunks(md_file_path, doc_name):
    """
    Parses extracted Markdown files into structured chunks based on section headers and tables.
    Enriches each chunk with document metadata for ChromaDB indexing.
    """
    if not os.path.exists(md_file_path):
        print(f"File not found: {md_file_path}")
        return []

    with open(md_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by major Markdown headings or HTML comment page markers
    raw_sections = re.split(r'\n(?=### |<!-- Page )', content)
    chunks = []

    for idx, sec in enumerate(raw_sections):
        sec_text = sec.strip()
        if not sec_text:
            continue
        
        # Extract page number if present
        page_match = re.search(r'<!-- Page (\d+) -->', sec_text)
        page_num = int(page_match.group(1)) if page_match else None

        # Determine chunk type
        is_table = "|\n|" in sec_text or "Extracted Table" in sec_text
        chunk_type = "table" if is_table else "text_clause"

        chunk_data = {
            "chunk_id": f"{doc_name}_chunk_{idx + 1:03d}",
            "text": sec_text,
            "metadata": {
                "source_doc": doc_name,
                "page_number": page_num,
                "chunk_type": chunk_type,
                "char_length": len(sec_text)
            }
        }
        chunks.append(chunk_data)

    return chunks

def run_chunking_pipeline():
    passages_dir = "data/extracted_passages"
    output_file = "data/processed_chunks.json"
    os.makedirs("data", exist_ok=True)

    all_chunks = []

    # Process Passage A
    passage_a_file = os.path.join(passages_dir, "passage_a_setbacks.md")
    chunks_a = parse_markdown_to_chunks(passage_a_file, doc_name="Passage_A_Setbacks")
    all_chunks.extend(chunks_a)

    # Process Passage B
    passage_b_file = os.path.join(passages_dir, "passage_b_road_width.md")
    chunks_b = parse_markdown_to_chunks(passage_b_file, doc_name="Passage_B_Road_Width")
    all_chunks.extend(chunks_b)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)

    print(f"Successfully processed {len(all_chunks)} chunks into: {output_file}")
    for c in all_chunks:
        print(f" - [{c['metadata']['source_doc']}] ID: {c['chunk_id']} | Type: {c['metadata']['chunk_type']} | Length: {c['metadata']['char_length']} chars")

if __name__ == "__main__":
    run_chunking_pipeline()