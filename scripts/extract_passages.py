import os
import pdfplumber
import pandas as pd

def extract_tables_and_text(pdf_path, page_numbers, output_md_path):
    """
    Extracts text and converts tables to Markdown format from specific PDF pages.
    """
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    full_content = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num in page_numbers:
            page = pdf.pages[page_num - 1]  # 1-indexed to 0-indexed
            
            # Extract standard text
            text = page.extract_text()
            if text:
                full_content.append(f"<!-- Page {page_num} -->\n" + text)

            # Extract tables if present
            tables = page.extract_tables()
            for idx, table in enumerate(tables):
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    # Clean up newlines within table cells
                    df = df.map(lambda x: str(x).replace('\n', ' ') if x else '')
                    markdown_table = df.to_markdown(index=False)
                    full_content.append(f"\n### Extracted Table {idx + 1} (Page {page_num})\n" + markdown_table)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_content))

    print(f" Successfully saved extracted passage to: {output_md_path}")

if __name__ == "__main__":
    # Adjust file paths and page numbers based on your corpus location
    CORPUS_DIR = "corpus"
    OUTPUT_DIR = "data/extracted_passages"

    bda_pdf = os.path.join(CORPUS_DIR, "BDA_RMP_2015_Zonal_Regulations.pdf")
    bbmp_pdf = os.path.join(CORPUS_DIR, "BBMP_Building_Byelaws_2003.pdf")

    # Example: Extract Passage A (Setbacks - Table 8/Table 4)
    if os.path.exists(bda_pdf):
        extract_tables_and_text(
            pdf_path=bda_pdf,
            page_numbers=[45, 46],  # Target pages containing Table 8
            output_md_path=os.path.join(OUTPUT_DIR, "passage_a_setbacks.md")
        )

    # Example: Extract Passage B (Road-width modifiers / Height limits)
    if os.path.exists(bbmp_pdf):
        extract_tables_and_text(
            pdf_path=bbmp_pdf,
            page_numbers=[22, 23],  # Target pages containing Clause 3.2.1 / 4.1.2
            output_md_path=os.path.join(OUTPUT_DIR, "passage_b_road_width.md")
        )