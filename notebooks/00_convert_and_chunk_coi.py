
#%%
import os
import re
import numpy as np
import json
import torch

from pyprojroot.here import here
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path

from langchain.docstore.document import Document
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk


from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, MarkdownFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

load_dotenv()
vdb_path = here() / "data/vdb"
doc_path = here() / "data/raw/coi_24_25.pdf"
output_path = here() / "data/processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

def replace_table_chunks(chunked_docs: List[DocChunk], tables: List[Document]) -> List[DocChunk]:
    """
    Account for inability to export chunked docling tables to Markdown. Replace with markdown tables.
    """
    updated_chunks = []
    table_refs = {table.self_ref: table for table in tables}  # Map table references
    appended_tables = set()  # Keep track of tables we've already appended
    
    for chunk in chunked_docs:
        text = chunk.text
        meta = chunk.meta
        doc_items = meta.doc_items

        # Check if the chunk corresponds to a table
        table_ref = next((item.self_ref for item in doc_items if item.self_ref in table_refs), None)
        
        if table_ref:
            # Skip if we've already appended this table
            if table_ref in appended_tables:
                continue
            # Else, chunk with properly formatted Markdown table
            markdown_table = table_refs[table_ref].export_to_markdown()
            updated_chunks.append(DocChunk(text=markdown_table, meta=meta))
            appended_tables.add(table_ref)
        else:
            # Keep the chunk as it is
            updated_chunks.append(chunk)
    
    return updated_chunks

#%%

# set up document converter - OCR capable, use cuda
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.do_cell_matching = True
pipeline_options.ocr_options.lang = ["es"]
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=8, device=AcceleratorDevice.AUTO
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        # , backend=MsWordDocumentBackend
        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
        InputFormat.MD: MarkdownFormatOption(pipeline_cls=SimplePipeline)
    },
    allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.IMAGE,
                        InputFormat.PPTX, InputFormat.MD]
)

# create tokenizer and chunker
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2",
    model_max_length=512,  # Enforce max token length
    truncation=True)

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512)


#%%

# convert and chunk the document
converted_doc = converter.convert(str(doc_path))
chunked_docs = list(chunker.chunk(converted_doc.document))
updated_chunked_docs = replace_table_chunks(chunked_docs=chunked_docs,
                                            tables=converted_doc.document.tables)


# %%

md_text = converted_doc.document.export_to_markdown()

output_file = str(here() / "data/raw/coi.md")

# Save markdown text to file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(md_text)

### ----IMPORTANT--- Update md file, adjust space and sys engr in appendix 2:
### Each should have major , \n\n (major abbr) \n\n "offered by the department of (major name)""

#%%

input_file = 'data/raw/coi.md'
output_file = 'data/raw/coi_updated.md'

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()

updated_lines = lines.copy()

# State dictionary to track our context.
state = {
    "inside_level2": False,
    "in_major_courses": False,
    "in_appendix_2": False,
    "dept_abbr": None
}

# Define keywords for key level-2 sections.
LEVEL2_KEYWORDS = [
    "course requirements",
    "supplemental information",
    "suggested course sequence",
    "course unit summary",
    "depth requirements"
]

def next_nonempty_line(i, lines):
    for j in range(i+1, len(lines)):
        if lines[j].strip():
            return lines[j]
    return ""

# Each rule is a function that takes (line, state, next_line, output) and returns a transformed line (or None).
def rule_main_header(line, state, next_line, output):
    if re.match(r'^#\s+[^#]', line):
        state["inside_level2"] = False
        state["in_major_courses"] = False
        return line
    return None

def rule_distribution_requirements(line, state, next_line, output):
    if line.strip().lower().startswith("distribution requirements"):
        state["inside_level2"] = True
        state["in_major_courses"] = False
        return f'## {line.strip()}\n'
    return None

def rule_key_header(line, state, next_line, output):
    # Try matching a header that starts with one or more '#' symbols.
    m = re.match(r'^#+\s*(.+)', line)
    if m:
        header = m.group(1).strip()
    else:
        # Fallback: check if the line exactly matches one of our key headers, e.g. "Supplemental Information:".
        header = None
        for kw in LEVEL2_KEYWORDS:
            pattern = fr'^\s*{kw.strip()}\s*:\s*$'
            if re.match(pattern, line, re.IGNORECASE):
                header = kw
                break
        if header is None:
            return None

    # Normalize header for checking by removing any leading label (like "C. ").
    header_clean = re.sub(r'^[A-Z]\.\s*', '', header, flags=re.IGNORECASE)
    # First, check if the header indicates a major's courses list.
    if re.search(r"major\s*'?s\s*courses", header_clean, re.IGNORECASE) or \
           (re.search(r"course sequence", header_clean, re.IGNORECASE) and re.search(r'\(\d+', header_clean)):
        state["in_major_courses"] = True
        state["inside_level2"] = True
        return f'### {header.title()}\n'
    # Otherwise, if the header contains one of the level-2 keywords or ends with "requirements:".
    elif any(kw in header_clean.lower() for kw in LEVEL2_KEYWORDS) or re.search(r'requirements\s*:?\s*$', header_clean, re.IGNORECASE):
        state["in_major_courses"] = False
        state["inside_level2"] = True
        if next_line and re.search(r'at\s+a\s+glance', next_line, re.IGNORECASE):
            return f'# {header.title()}\n'
        else:
            return f'## {header.title()}\n'
    else:
        # Not a key header—if we're inside a key section and not in a major courses block, use level-3.
        if state.get("inside_level2") and not state.get("in_major_courses"):
            return f'### {header.title()}\n'
        return f'## {header.title()}\n'

def rule_major_courses_verbatim(line, state, next_line, output):
    if state.get("in_major_courses"):
        return line  # Leave unchanged.
    return None

def rule_sub_label(line, state, next_line, output):
    if state.get("inside_level2") and not state.get("in_major_courses"):
        m = re.match(r'^\s*(?:-|##|\*)?\s*(([A-Z]|\d+)\.\s+.+)', line)
        if m:
            return f'### {m.group(1).strip()}\n'
    return None

def rule_chapter(line, state, next_line, output):
    m = re.match(r'^##\s*(CHAPTER\s+\d+)', line, re.IGNORECASE)
    if m:
        return f'# {m.group(1).strip()}\n'
    return None

def rule_section(line, state, next_line, output):
    m = re.match(r'^##\s*(SECTION\s+\d+-\d+)', line, re.IGNORECASE)
    if m:
        return f'## {m.group(1).strip()}\n'
    return None

def rule_numeric_subheader(line, state, next_line, output):
    m = re.match(r'^##\s*(\d+-\d+(?:\.\d+)*)', line)
    if m:
        return f'### {m.group(1).strip()}\n'
    return None

def rule_at_a_glance(line, state, next_line, output):
    if re.search(r'at\s+a\s+glance', line, re.IGNORECASE):
        # Upgrade the previous header in output to level-1.
        for j in range(len(output)-1, -1, -1):
            if output[j].strip() == "":
                continue
            if re.match(r'^(##\s+|###\s+)', output[j]):
                output[j] = re.sub(r'^(##\s+|###\s+)', '# ', output[j])
                break
        return line
    return None

# --- rules for Appendix 2 and Course Descriptions ---
def rule_appendix_header(line, state, next_line, output):
    # If the line is a header matching "Appendix 2: Course Descriptions"
    if re.match(r'^#*\s*(Appendix\s+2\s*):', line, re.IGNORECASE):
        state["in_appendix_2"] = True
        # Reset any previous department abbreviation.
        state["dept_abbr"] = None
        return f'# {line.lstrip("#").strip()}\n'
    return None

def rule_capture_dept_abbr(line, state, next_line, output):
    # If the line contains "Offered by the Department of", then look backwards.
    if re.search(r'^Offered by', line.strip(), re.IGNORECASE):
        # Look back over the last 3 nonempty output lines.
        for prev_line in reversed([l for l in output if l.strip()]):
            m = re.search(r'\(([^)]+)\)', prev_line)
            if m:
                state["dept_abbr"] = m.group(1).strip()
                break
        return line  # Return the "Offered by" line unchanged.
    return None

def rule_course_description_divider(line, state, next_line, output):
    # If we are in Appendix 2 and we have captured a department abbreviation,
    # then if the line starts with that abbreviation followed by a space and a digit,
    # convert it to a level-4 header.
    if state.get("in_appendix_2") and state.get("dept_abbr"):
        abbr = state["dept_abbr"]
        normalized_line = re.sub(r'\s+', ' ', line).strip()
        pattern = fr'^{re.escape(abbr)}\s+[0-9A-Za-z]'
        if re.match(pattern, normalized_line):
            return f'#### {normalized_line}\n'
    return None

def rule_default(line, state, next_line, output):
    return line

# List of rules in order.
rules = [
    rule_appendix_header,
    rule_main_header,
    rule_distribution_requirements,
    rule_key_header,
    rule_major_courses_verbatim,
    rule_sub_label,
    rule_chapter,
    rule_section,
    rule_numeric_subheader,
    rule_at_a_glance,
    rule_capture_dept_abbr,
    rule_course_description_divider,
    rule_default
]

# Process each line through the rule pipeline.
output_lines = []
i = 0
while i < len(lines):
    current_line = lines[i]
    nxt = next_nonempty_line(i, lines)
    transformed = None
    # Try each rule in order; the first that returns non-None is used.
    for rule in rules:
        result = rule(current_line, state, nxt, output_lines)
        if result is not None:
            transformed = result
            break
    output_lines.append(transformed)
    i += 1

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

#%%

# convert other documents

for doc_path in Path.iterdir(here() / "data/raw/"):
    if doc_path.suffix == ".pdf" and not 'coi' in doc_path.stem.lower():

        print(f"Processing {doc_path}...")
        output_filename = output_path / f"{doc_path.stem}.md"
        # convert and chunk the document
        converted_doc = converter.convert(str(doc_path))
        md_text = converted_doc.document.export_to_markdown()

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(md_text)

