# extract_1gb_evals.py
import json
import zstandard as zstd
from io import TextIOWrapper
from tqdm import tqdm
import math
import os
import time

# --- Configuration ---
# !! UPDATE THESE PATHS !!
INPUT_ZST_FILE = r"C:/Users/admin/Desktop/Chess_project/fen_data/lichess_db_eval.jsonl.zst"
OUTPUT_TSV_FILE = r"C:/Users/admin/Desktop/Chess_project/fen_data/first_1.5GB_evaluations.tsv" # Output for 1GB data

# --- Limit ---
BYTES_TO_PROCESS = 1.5 * 1024 * 1024 * 1024 # Approximately 1 GB
# -------------

def extract_first_gb():
    """
    Reads approx the first 1GB of a Zstandard compressed JSONL file,
    extracts the highest-depth evaluation for each FEN (no filtering),
    and writes the FEN and evaluation to a TSV file.
    """
    start_time = time.time()
    count_lines_read = 0
    count_kept = 0
    count_skipped_other = 0
    bytes_read_approx = 0

    print(f"Starting extraction (first ~1GB):")
    print(f"Input: {INPUT_ZST_FILE}")
    print(f"Output: {OUTPUT_TSV_FILE}")
    print(f"Processing limit: {BYTES_TO_PROCESS / 1e9:.2f} GB")

    output_dir = os.path.dirname(OUTPUT_TSV_FILE)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output dir: {e}")
            return

    dctx = zstd.ZstdDecompressor()
    file_handle = None
    stream_context = None
    text_stream = None
    outfile_handle = None
    pbar = None

    try:
        total_input_size = os.path.getsize(INPUT_ZST_FILE) if os.path.exists(INPUT_ZST_FILE) else 0
        if total_input_size == 0:
            print(f"Error: Input file empty/not found.")
            return
        actual_bytes_to_process = min(BYTES_TO_PROCESS, total_input_size)

        file_handle = open(INPUT_ZST_FILE, 'rb')
        stream_context = dctx.stream_reader(file_handle)
        text_stream = TextIOWrapper(stream_context, encoding='utf-8', errors='ignore')
        outfile_handle = open(OUTPUT_TSV_FILE, 'w', encoding='utf-8')
        outfile_handle.write("fen\tevaluation\n") # Write header

        print("Starting decompression and extraction...")
        pbar = tqdm(total=actual_bytes_to_process, unit='B', unit_scale=True, desc="Extracting 1GB", mininterval=1.0)
        last_update_pos = 0

        for line in text_stream:
            count_lines_read += 1
            try:
                current_pos = file_handle.tell()
                bytes_read_approx = current_pos
                if current_pos > last_update_pos:
                    pbar.update(current_pos - last_update_pos)
                    last_update_pos = current_pos
                if bytes_read_approx >= actual_bytes_to_process:
                    print(f"\nReached ~{actual_bytes_to_process / 1e9:.2f} GB read limit. Stopping.")
                    break

                sample = json.loads(line)
                fen_string = sample.get('fen')
                evals = sample.get('evals')

                if not fen_string or not isinstance(evals, list) or not evals:
                    count_skipped_other += 1
                    continue

                try:
                    best_eval_info = max(evals, key=lambda e: e.get('depth', -1))
                except (TypeError, ValueError):
                    count_skipped_other += 1
                    continue

                pvs = best_eval_info.get('pvs')
                if not isinstance(pvs, list) or not pvs:
                    count_skipped_other += 1
                    continue
                
                top_pv = pvs[0]
                if not isinstance(top_pv, dict):
                    count_skipped_other += 1
                    continue

                if 'cp' in top_pv:
                    evaluation = top_pv['cp']
                    if not isinstance(evaluation, (int, float)): count_skipped_other += 1; continue
                elif 'mate' in top_pv:
                    mate_in = top_pv['mate']
                    if not isinstance(mate_in, (int, float)): count_skipped_other += 1; continue
                    evaluation = 15000 * math.copysign(1, mate_in)
                else:
                    count_skipped_other += 1; continue

                clean_fen = fen_string.replace('\t', ' ').replace('\n', ' ')
                outfile_handle.write(f"{clean_fen}\t{int(evaluation)}\n")
                count_kept += 1

            except json.JSONDecodeError: count_skipped_other += 1; continue
            except Exception as e: count_skipped_other += 1; continue

        if pbar:
            if file_handle and not file_handle.closed:
                try:
                    current_pos = file_handle.tell()
                    if current_pos > last_update_pos:
                        pbar.update(current_pos - last_update_pos)
                except ValueError:
                    pass
            if pbar.n < pbar.total:
                pbar.n = pbar.total
                pbar.refresh()
            pbar.close()
        print("\nExtraction finished.")

    except Exception as e:
        print(f"\nError: {e}")
        if pbar:
            pbar.close()
    finally:
        print("\nClosing resources...")
        if outfile_handle:
            try:
                outfile_handle.close()
            except Exception as e:
                print(f"  (Ignored error closing output file: {e})")
        if text_stream:
            try:
                text_stream.close()
            except Exception as e:
                print(f"  (Ignored error closing text stream: {e})")
        if stream_context:
            try:
                stream_context.close()
            except Exception as e:
                print(f"  (Ignored error closing zstd stream: {e}")
        if file_handle:
            try:
                file_handle.close()
            except Exception as e:
                print(f"  (Ignored error closing input file handle: {e})")
        print("Resources closed.")

    end_time = time.time(); duration = end_time - start_time
    print("\n--- Extraction Summary ---")
    print(f"Approx compressed data processed: {bytes_read_approx / 1e9:.2f} GB")
    print(f"Total lines read: {count_lines_read:,}")
    print(f"Positions kept (highest depth eval): {count_kept:,}")
    print(f"Skipped (other issues): {count_skipped_other:,}")
    print(f"Extracted data saved to: {OUTPUT_TSV_FILE}")
    print(f"Extraction took {duration / 60:.2f} minutes.")

if __name__ == "__main__":
    if INPUT_ZST_FILE.startswith("path/to/your"):
        print("!!! Please update the INPUT_ZST_FILE and OUTPUT_TSV_FILE paths !!!")
        exit(1)
    
    extract_first_gb()