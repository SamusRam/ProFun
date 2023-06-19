"""Utils for MSA (Multiple Sequence Alignment)"""
import os
from typing import Iterable, Optional


def get_fasta_seqs(seqs: Iterable[str], ids: Optional[Iterable[str]] = None):
    """
    This function dumps list of sequences with optional ids to a fasta file
    """
    if ids is None:
        full_entries = [
            f">{i}\n{''.join(tps_seq.split())}" for i, tps_seq in enumerate(seqs)
        ]
    else:
        full_entries = [
            f">{id}\n{''.join(tps_seq.split())}" for id, tps_seq in zip(ids, seqs)
        ]
    return "\n".join(full_entries)


def generate_msa_mafft(*, seqs: Optional[Iterable[str]] = None, ids: Iterable[str] = None,
                       fasta_str: str = None,
                       output_name: str = "_msa.fasta",
                       n_jobs: int = 26,
                       clustal_output_format: bool = True):
    """
    This function generates multiple sequence alignment using MAFFT

    Either accepts all fasta-format sequences prepared in `fasta_str` argument,
    or prepares the fasta-format sequences based on
    """
    assert fasta_str is None or seqs is None, ("The input sequences must be passed either as an iterable of strings or "
                                               "as the preprocessed fasta_str, "
                                               "but cannot be passed by both options simultaneously")
    fasta_str = get_fasta_seqs(seqs, ids)
    with open("_temp_mafft.fasta", "w", encoding="utf8") as f:
        f.writelines(fasta_str.replace("'", "").replace('"', ""))
    os.system(
        f"mafft --thread {n_jobs} --auto --quiet {'--clustalout ' if clustal_output_format else ''}_temp_mafft.fasta > {output_name}")
    os.remove("_temp_mafft.fasta")
