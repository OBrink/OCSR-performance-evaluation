import csv
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from typing import List

"""
Achim's description of the CES calculation:

1. Determine number of invalid predicted structures N(invalid) and evaluate Tanimoto similarity of all valid predicted structures in interval [0, 1]
2. Count structures with Tanimoto similarities
   - rounded to 0.00 (= N(0.00)) and correct with N(invalid), i.e. N(0.00, new) = N(0.00, old) + N(invalid)
   - rounded to 0.01 (= N(0.01))
   - rounded to 0.02 (= N(0.02))
   ...
   - rounded to 1.00 (= N(1.00))
   Results are 101 numbers N(i)
3. Multiply each N(i) with a penalty factor to get individual error score E(i):
   E(0.00) = N(0.00, new) * 100
   E(0.01) = N(0.01)      *  99
   E(0.02) = N(0.02)      *  98
   ...
   E(1.00) = N(1.00)      *   0
4. Calculate sum of all 101 error scores E(0.00) to E(1.00) to arrive at the cumulated error score CES: This is a number between 0 and 100 where
   - CES =   0: Best predictive result with all molecule pairs having Tanimoto similarity 1
   - CES = 100: Worst predictive result with all molecule pairs having Tanimoto similarity 0 (or invalid)
   - Equal distribution leads to CES = 50 ("perfect mean")
   - Linear progressive distribution leads to CES = 33 ("below mean")
   - Linear regressive distribution leads to CES = 67 ("above mean")
   """


def _get_delimiter(
    csv_path: str
):
    """
    Given a a path of a csv file, return the delimiter used in the file
    or None if no delimiter is found.

    This function is based on code from
    https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/utils.py

    Args:
        csv_path (str): Path of CSV file

    Returns:
        str, None: Delimiter used in CSV file or None if no delimiter is found
    """
    with open(csv_path, 'r') as input_file:
        lines = input_file.readlines()
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(lines[-1])
        delimiter = dialect.delimiter
        expected_delimiters = ['\t', ',', ';', ' ', '  ', '   ', '    ', '|']
        if delimiter in expected_delimiters:
            return delimiter
        else:
            return


def _get_smiles_column(
    smilesfile_path: str,
    delimiter: str
):
    """
    Given a path of a smiles file and a delimiter, return the index of the column that
    contains the SMILES or None if no SMILES are found.

    This function is based on code from
    https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/utils.py

    Args:
        smilesfile_path (str): Path of SMILES file
        delimiter (str): delimiter used in SMILES file.
                         Set to " " if no delimiter is used.

    Raises:
        ValueError: If no valid SMILES strings are found in the file

    Returns:
        int, None: column index or None if no SMILES are found
    """
    with open(smilesfile_path, 'r', newline='') as input_file:
        rows = csv.reader(input_file, delimiter=delimiter)
        rows = list(rows)
        for row in rows[::-1]:
            with rdBase.BlockLogs() as _:
                for i, smi in enumerate(row):
                    if Chem.MolFromSmiles(smi, sanitize=False) is not None:
                        return i
        raise ValueError('No valid SMILES found in file.')


def _has_header(smilesfile_path: str, delimiter: str) -> bool:
    """
    Given a path of a smiles file and a delimiter, return True if the file has a header.
    This is simply done by checking if the first line contains a valid SMILES string,
    so it is not a very robust method, but it should work for most cases with the worst
    case that we lose one molecule (which should not affect the final results too much).

    This function is based on code from
    https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/utils.py

    Args:
        smilesfile_path (str): path of SMILES file
        delimiter (str): delimiter used in SMILES file.
                         Set to " " if no delimiter is used.

    Returns:
        bool: True if the file has a header, False otherwise
    """
    with open(smilesfile_path, 'r') as input_file:
        header_line = input_file.readline()[:-1]
        elements = header_line.split(delimiter)
        with rdBase.BlockLogs() as _:
            for element in elements:
                if Chem.MolFromSmiles(element, sanitize=False):
                    return False
        return True


def get_mols_from_smilesfile(
    smilesfile_path: str,
) -> Chem.RDMolSupplier:
    """
    Get RDMolSupplier object from SMILES file.

    Args:
        smilesfile_path (str): Path of SMILES file

    Returns:
        Chem.RDMolSupplier: RDMolSupplier object
    """
    delimiter = _get_delimiter(smilesfile_path)
    if not delimiter:
        col = 0
        delimiter = ' '
    else:
        col = _get_smiles_column(smilesfile_path, delimiter)
    header_exists = _has_header(smilesfile=smilesfile_path, delimiter=delimiter)
    if not header_exists:
        with open(smilesfile_path, 'r') as input_file:
            content = 'DUMMY_HEADER\n' + input_file.read()
        mols = Chem.SmilesMolSupplierFromText(content,
                                              delimiter=delimiter,
                                              smilesColumn=col)
    else:
        mols = Chem.SmilesMolSupplier(smilesfile_path,
                                      delimiter=delimiter,
                                      smilesColumn=col)
    return mols


def calculate_similarity_group_counts(
    similarities: List[float],
    bin_size: float = 0.01
) -> np.ndarray:
    """
    Group similarities into bins of size bin_size. Given a list of float numbers and
    a bin size, return a numpy array of counts of the number of similarities that fall
    into each bin.

    Args:
        similarities (List[float]): list of similarity values (float)
        bin_size (float, optional): bin size. Defaults to 0.01.

    Returns:
        np.ndarray: number of similarity values that fall into each bin
    """
    similarities = [round(similarity, 2) for similarity in similarities]
    bins = np.arange(0, 1 + bin_size, bin_size)
    return np.histogram(similarities, bins=bins)[0]
