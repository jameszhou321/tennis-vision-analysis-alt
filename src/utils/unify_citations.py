"""
Global Unified Citation Numbering Script.
1. Collect references across all thesis chapters
2. Deduplicate and merge references
3. Update all in-text citation marker flags
Usage: cd ProjectAnnotationAndTesting && .venv/Scripts/python src/utils/unify_citations.py
"""
import re, os, sys

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PAPER_DIR = os.path.join(os.path.dirname(_PROJECT_DIR), "Paper")  # Thesis directory is outside the project root

CHAPTERS = [
    "Chapter_1_Introduction.md",
    "Chapter_2_Related_Technical_Fundamentals.md",
    "Chapter_3_Dataset_Construction_and_Annotation_Toolchain.md",
    "Chapter_4_Data_Preprocessing_and_Multimodal_Feature_Extraction.md",
    "Chapter_5_MSTFormer_Multi_Stream_Action_Recognition_Model.md",
    "Chapter_6_Experiments_and_Analysis.md",
    "Chapter_7_System_Integration_and_Demo_Application.md",
    "Chapter_8_Conclusion_and_Future_Work+References.md",
]

def parse_references(text):
    """Extract reference list elements from text content."""
    refs = []
    in_refs = False
    for line in text.split('\n'):
        if '## References' in line or '# References' in line:
            in_refs = True
            continue
        if in_refs:
            m = re.match(r'^\[(\d+)\]\s+(.+)', line)
            if m:
                refs.append((int(m.group(1)), m.group(2).strip()))
            elif line.strip() == '':
                pass
            elif not line.startswith('['):
                # End of reference section
                if refs:  # Stop evaluating only after collection sequence has initialized
                    break
    return refs

def normalize_ref(text):
    """Normalize reference text strings to perform precise deduplication matching."""
    t = text.lower().strip()
    # Strip punctuation marks and extra spacing layout markers
    t = re.sub(r'[.,:;\-\[\]\(\)""\'\']', '', t)
    t = re.sub(r'\s+', ' ', t)
    # Extract structural components using the first 80 chars as matching criteria hash key
    return t[:80]

def main():
    all_chapter_refs = {}
    seen_refs = {}  # normalized_text -> (global_id, full_text)
    global_refs = []
    chapter_mappings = {}  # chapter -> {old_num -> global_num}

    # 1. Read all chapter assets from disk
    for ch in CHAPTERS:
        path = os.path.join(_PAPER_DIR, ch)
        if not os.path.exists(path):
            print(f"!!  {ch} does not exist, skipping asset entry.")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        refs = parse_references(text)
        all_chapter_refs[ch] = (text, refs)
        print(f"[INFO] {ch}: {len(refs)} reference entries found")

    # 2. Perform global reference list deduplication and reconstruction
    for ch, (text, refs) in all_chapter_refs.items():
        mapping = {}
        for old_num, full_text in refs:
            key = normalize_ref(full_text)
            if key in seen_refs:
                global_id, _ = seen_refs[key]
            else:
                global_id = len(global_refs) + 1
                global_refs.append((global_id, full_text))
                seen_refs[key] = (global_id, full_text)
            mapping[old_num] = global_id
        chapter_mappings[ch] = mapping

    print(f"\n[DATA] Total unique references after global deduplication: {len(global_refs)}")
    print(f"   Sum total before deduplication: {sum(len(r) for _,r in all_chapter_refs.values())} entries")
    print(f"   Deduplicated entries removed: {sum(len(r) for _,r in all_chapter_refs.values()) - len(global_refs)} entries\n")

    # 3. Update all in-text citation marker elements and append global references block
    for ch, (text, refs) in all_chapter_refs.items():
        mapping = chapter_mappings[ch]

        # Replace in-text [N] citation markers
        # Matches [number] but excludes lines beginning with [number] (the reference definitions themselves)
        def replace_ref(m):
            num = int(m.group(1))
            if num in mapping:
                return f"[{mapping[num]}]"
            return m.group(0)

        # Isolate text body segment from original reference lists
        parts = re.split(r'(## References|# References)', text)
        if len(parts) > 1:
            body = parts[0]
            ref_section = ''.join(parts[1:])
        else:
            body = text
            ref_section = ''

        # Replace citation markers in the body context
        new_body = re.sub(r'\[(\d+)\]', replace_ref, body)

        # Generate subset of global reference items specifically utilized within this chapter body
        used_global_ids = set(mapping.values())
        global_ref_text = '\n'.join(
            f'[{gid}] {text}' for gid, text in global_refs
            if gid in used_global_ids
        )

        # Format and construct unified reference block segments
        if ref_section:
            new_ref_section = '\n\n## References (Globally Unified Numbering)\n\n' + global_ref_text + '\n'
        else:
            new_ref_section = '\n\n## References (Globally Unified Numbering)\n\n' + global_ref_text + '\n'

        new_text = new_body + new_ref_section

        # Persist updated content back to disk source target file paths
        out_path = os.path.join(_PAPER_DIR, ch)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(new_text)

        print(f"[OK] {ch}: Citation numbering mapping complete ({len(refs)}→{len(used_global_ids)} entries)")

    # 4. Export complete global reference lists data output structure metrics
    print(f"\n{'='*60}")
    print(f"Global Reference Bibliography Matrix ({len(global_refs)} entries)")
    print(f"{'='*60}")
    for gid, text in global_refs:
        print(f"[{gid}] {text}")


if __name__ == "__main__":
    main()