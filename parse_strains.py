import re

INPUT_FILE = "strainnames.txt"
OUTPUT_FILE = "strainnames_cleaned.csv"

ACCEPTABLE_STRAINS = {"alpha", "beta", "delta", "gamma", "omicron"}

with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

results = []

for i, line in enumerate(lines, 1):
    line = line.strip()
    if not line:
        continue
    
    sequence_id = None
    strain = None
    
    if line.startswith("Toggle selection of sequence:"):

        parts = re.split(r'[\t\s]+', line)
        
        # Find sequence ID 
        for part in parts:
            if "." in part and len(part) > 5: 
                sequence_id = part
                break
        
        # Search the entire line for acceptable strain names
        line_lower = line.lower()
        for strain_name in ACCEPTABLE_STRAINS:
            if strain_name in line_lower:
                strain = strain_name
                break
    else:

        if "," in line:
            parts = line.split(",", 1)  
            sequence_id = parts[0].strip()
            strain_raw = parts[1].strip() if len(parts) > 1 else None
            if strain_raw:
                strain = strain_raw.lower()
    
    # Only keep if both sequence ID and acceptable strain found
    if sequence_id and strain and strain in ACCEPTABLE_STRAINS:
        results.append(f"{sequence_id},{strain}")
        print(f"Line {i}: Added {sequence_id},{strain}")
    elif sequence_id and strain:
        print(f"Line {i}: Rejected {sequence_id},{strain} (not in acceptable list)")


with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    outfile.write("\n".join(results))

print(f"\nWrote {len(results)} filtered sequences to {OUTPUT_FILE}")
