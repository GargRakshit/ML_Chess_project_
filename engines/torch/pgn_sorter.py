import os
import subprocess
import multiprocessing
from tqdm import tqdm
import platform

# --- Configuration ---
INPUT_DIR = r"C:\Users\admin\Desktop\Chess_project\new_data"
OUTPUT_DIR = r"C:\Users\admin\Desktop\Chess_project\filtered_output"
MIN_ELO = 2500
WORKERS = max(1, multiprocessing.cpu_count() - 1)

# --- Paths to Executables ---
ZSTD_EXE = "zstd.exe"
PGN_EXTRACT_EXE = "pgn-extract.exe"

# --- Globals for shared resources ---
manager = None
pbar = None

def init_globals(p_bar_global):
    global pbar
    pbar = p_bar_global

def process_file_with_external_tools(filepath):
    input_filename = os.path.basename(filepath)
    output_filename = input_filename.replace(".pgn.zst", f"_elo{MIN_ELO}.pgn")
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    zstd_args = [ZSTD_EXE, "-dqc", filepath]
    pgn_extract_args = [
        PGN_EXTRACT_EXE,
        "-e", str(MIN_ELO),
        "-E", str(MIN_ELO),
        "-o", output_path
    ]

    result_message = f"Processing {input_filename}..."
    zstd_proc = None
    pgn_proc = None
    file_size = 0

    try:
        file_size = os.path.getsize(filepath)

        zstd_proc = subprocess.Popen(
            zstd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )

        pgn_proc = subprocess.Popen(
            pgn_extract_args,
            stdin=zstd_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )

        # Do NOT close zstd_proc.stdout manually; communicate() will handle it
        pgn_stdout, pgn_stderr = pgn_proc.communicate()
        pgn_returncode = pgn_proc.returncode

        zstd_stdout, zstd_stderr = zstd_proc.communicate()
        zstd_returncode = zstd_proc.returncode

        if pgn_returncode != 0:
            pgn_stderr_str = pgn_stderr.decode().strip()
            if "Unable to find" in pgn_stderr_str:
                result_message = f"⚠️ No games ≥ {MIN_ELO} found in {input_filename}."
            else:
                error_details = pgn_stderr_str or zstd_stderr.decode().strip() or "Unknown pgn-extract error"
                raise subprocess.CalledProcessError(pgn_returncode, pgn_extract_args, output=pgn_stdout, stderr=pgn_stderr)

        if zstd_returncode != 0:
            zstd_stderr_str = zstd_stderr.decode().strip()
            if "Broken pipe" not in zstd_stderr_str and zstd_stderr_str:
                print(f"\nWarning: Zstd process for {input_filename} exited with code {zstd_returncode}. Stderr: '{zstd_stderr_str[:200]}'")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            result_message = f"✅ Successfully processed {input_filename}"
        elif os.path.exists(output_path) and os.path.getsize(output_path) == 0:
            try:
                os.remove(output_path)
            except OSError:
                pass
            if "⚠️" not in result_message:
                result_message = f"⚠️ Processed {input_filename}, but no games met criteria (output empty)."
        else:
            result_message = f"❌ Processed {input_filename}, but output file not created/empty. PGN Extract Stdout: '{pgn_stdout.decode().strip()[:200]}'"

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode().strip() if hasattr(e.stderr, 'decode') else str(e.stderr)
        stdout_output = e.stdout.decode().strip() if hasattr(e.stdout, 'decode') else str(e.stdout)
        result_message = f"❌ Error processing {input_filename}: Command '{' '.join(e.cmd)}' failed with exit code {e.returncode}. Stderr: '{stderr_output[:200]}'. Stdout: '{stdout_output[:200]}'"
    except FileNotFoundError as e:
        result_message = f"❌ Error: Executable not found ({e.filename}). Check paths."
    except Exception as e:
        result_message = f"❌ Unexpected error processing {input_filename}: {type(e).__name__}: {e}"
    finally:
        if pgn_proc and pgn_proc.poll() is None:
            pgn_proc.kill()
            pgn_proc.communicate()
        if zstd_proc and zstd_proc.poll() is None:
            zstd_proc.kill()
            zstd_proc.communicate()

    return result_message, file_size

def update_progress(result):
    status_message, size_processed = result
    print(status_message)
    if pbar:
        pbar.update(size_processed)

def main():
    global manager, pbar

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".pgn.zst")]
        if not files:
            print(f"Error: No .pgn.zst files found in '{INPUT_DIR}'.")
            return
        total_size = sum(os.path.getsize(f) for f in files)
    except FileNotFoundError:
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    print(f"Found {len(files)} files ({total_size / 1e9:.2f} GB). Starting processing using {WORKERS} workers...")
    print(f"Executables:\n  ZSTD: {ZSTD_EXE}\n  PGN-Extract: {PGN_EXTRACT_EXE}")

    manager = multiprocessing.Manager()
    pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Overall Progress")

    with multiprocessing.Pool(WORKERS) as pool:
        async_results = [pool.apply_async(process_file_with_external_tools, args=(f,), callback=update_progress) for f in files]
        pool.close()
        pool.join()

    pbar.close()

    final_statuses = [res.get()[0] for res in async_results]
    errors = [s for s in final_statuses if "❌ Error" in s]
    success_count = len(files) - len(errors)

    print(f"\n--- Processing Summary ---")
    print(f"Successfully processed files (created non-empty output): {success_count}")
    if errors:
        print("\n--- Issues Encountered ---")
        for issue in errors:
            print(issue)

    print(f"\n--- All Processing Complete ---")
    print(f"Filtered files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
