"""Script for handling the parsing of PDF files.

This script contains functions to:
- Create directories if they don't exist.
- Run the Nougat tool asynchronously to parse PDF files.

However, this file is not currently used within the pipeline.
"""

import subprocess
from pathlib import Path


def create_folder(directory: Path) -> None:
    """Ensures the folder exists, creating it if necessary.

    Args:
        directory (Path): The path of the directory to be created.
    """
    if not directory.exists():
        directory.mkdir(parents=True)


def run_nougat(
        input_file: Path, 
        output_dir: Path, 
        small_model: bool = False
    ) -> subprocess.Popen:
    """Asynchronously runs Nougat on a single file, creating a .mmd file.

    Args:
        input_file (Path): Path to the PDF document to be parsed.
        output_dir (Path): Path to the directory to save the .mmd file.
        small_model (bool, optional): Whether to use the less resource-intensive small model. Defaults to False.

    Returns:
        subprocess.Popen: The created process for future access if needed.
    """
    # Select model name for the command
    model_name = "0.1.0-small" if small_model else "0.1.0-base"

    # Calculate and run command using a safer subprocess call
    command = [
        "nougat",
        str(input_file),
        "-o",
        str(output_dir),
        "-m",
        model_name,
    ]
    print("Running command:", " ".join(command))
    try:
        return subprocess.Popen(command)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            """The 'nougat' command is not found. 
            Ensure it is installed and in your PATH.
        """) from err
    except Exception as err:
        raise RuntimeError(
            f"""Error occurred while running the command: {err}
        """) from err


def main() -> None:
    """Main entry point for managing the parsing of the PDF's."""
    # Path to current directory
    cwd = Path.cwd()
    print("CWD:", cwd)

    # Path to scraped files
    input_file_path = cwd / "data" / "NY_DPS_Documents"

    # Path to output
    output_file_path = cwd / "mmds"
    create_folder(output_file_path)

    # List running processes
    processes = []

    # Loop through files
    for input_file_name in input_file_path.iterdir():
        if input_file_name.is_file():
            # Get file name for output .mmd file
            output_file_name = input_file_name.stem + ".mmd"

            # Get expected output file
            output_file = output_file_path / output_file_name

            # Verify file does not already exist to avoid wasted resources
            if not output_file.exists():
                # Run Nougat on file
                process = run_nougat(input_file_name, output_file_path, False)
                processes.append(process)
                print("Finished with:", output_file)


if __name__ == "__main__":
    main()
