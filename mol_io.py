import csv
from rdkit import Chem
from rdkit import rdBase


def _get_mols_from_smilesfile(
    smilesfile_path: str,
) -> Chem.SmilesMolSupplier:
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
    header_exists = _has_header(smilesfile_path=smilesfile_path, delimiter=delimiter)
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
