"""This script downloads protein structures predicted by AlphaFold2"""

import argparse
import logging
from pathlib import Path

import requests

import pandas as pd  # type: ignore

from tqdm.auto import tqdm

logging.basicConfig()
logger = logging.getLogger("Downloading AlphaFold2 structures")
logger.setLevel(logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """
    This function parses arguments
    :return: current argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structures-output-path", type=str, default="../data/af_structs"
    )
    parser.add_argument("--path-to-file-with-ids", type=str, 
                        default="../data/uniprot_ids_of_interest.txt", help="Path to a file containing UniProt IDs,"
                                                                            "for which the script will download AF2 structures")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cl_args = parse_args()
    root_path = cl_args.input_root_path

    root_af_cafa = Path(cl_args.structures_output_path)
    if not root_af_cafa.exists():
        root_af_cafa.mkdir()

    with open(cl_args.path_to_file_with_ids, 'r') as file:
        all_ids_of_interest = [line.strip() for line in file.readlines()]

    for uniprot_id in tqdm(all_ids_of_interest):
        try:
            URL = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v3.pdb"
            response = requests.get(URL)
            with open(root_af_cafa / f"{uniprot_id}.pdb", "wb") as file:
                file.write(response.content)
        except:
            logger.warning(f"Error downloading AlphaFold2 structure for {uniprot_id}")
            continue