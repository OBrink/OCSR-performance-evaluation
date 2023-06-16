from cumulative_error_score import (calculate_similarity_group_counts,
                                    calculate_cumulative_error_score)
from mol_io import (_get_delimiter, _get_smiles_column, _has_header,
                    _get_mols_from_smilesfile)
import numpy as np
import unittest


class TestCesCalculation(unittest.TestCase):
    def test_calculate_cumulative_error_score(self,):
        # Standard cases
        self.assertEqual(calculate_cumulative_error_score(np.zeros(100)), 100)
        self.assertEqual(calculate_cumulative_error_score(np.zeros(1000)), 100)
        self.assertEqual(calculate_cumulative_error_score(np.ones(100)), 0)
        self.assertEqual(calculate_cumulative_error_score(np.ones(1000)), 0)
        self.assertEqual(calculate_cumulative_error_score(np.arange(0, 1.01, 0.01)), 50)
        # Progressive distribution
        sim_arr = []
        for index in range(1001):
            for _ in range(index):
                sim_arr.append(np.arange(0, 1.001, 0.001)[index])
        self.assertEqual(round(calculate_cumulative_error_score(sim_arr), 1), 33.3)
        # Regressive distribution
        sim_arr = []
        for index in range(1001):
            for _ in range(index):
                sim_arr.append(round(np.arange(1, -0.001, -0.001)[index], 3))
            
        self.assertEqual(round(calculate_cumulative_error_score(sim_arr), 1), 66.7)
        # ValueError if similarity values are not between 0 and 1
        self.assertRaises(ValueError, calculate_similarity_group_counts, [-1, 0, 0])
        self.assertRaises(ValueError, calculate_similarity_group_counts, [1.01, 0, 0])

    def test_calculate_similarity_group_counts(self,):
        similarities = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected = np.array([
            5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9])
        assert np.array_equal(calculate_similarity_group_counts(similarities), expected)
        self.assertRaises(ValueError, calculate_similarity_group_counts, [-1, 0, 0])
        self.assertRaises(ValueError, calculate_similarity_group_counts, [1.01, 0, 0])

    def test_get_delimiter(self):
        # Origin:
        # https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/test_lwreg.py
        self.assertEqual(
            _get_delimiter('test_data/test_smiles_no_delim.smi'), None)
        self.assertEqual(
            _get_delimiter(
                'test_data/test_smiles_no_delim_with_header.smi'), None)
        self.assertEqual(
            _get_delimiter('test_data/test_smiles_with_header.smi'), ';')
        self.assertEqual(_get_delimiter('test_data/test_smiles.smi'),
                         ' ')

    def test_get_smiles_column(self):
        # Origin:
        # https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/test_lwreg.py
        self.assertEqual(
            _get_smiles_column('test_data/test_smiles_with_header.smi',
                               delimiter=';'), 1)
        self.assertEqual(
            _get_smiles_column('test_data/test_smiles.smi',
                               delimiter=' '), 1)

    def test_has_header(self):
        # Origin:
        # https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/test_lwreg.py
        self.assertTrue(
            _has_header('test_data/test_smiles_no_delim_with_header.smi',
                        delimiter=' '))
        self.assertFalse(
            _has_header('test_data/test_smiles_no_delim.smi',
                        delimiter=' '))
        self.assertTrue(
            _has_header('test_data/test_smiles_with_header.smi',
                        delimiter=','))
        self.assertFalse(
            _has_header('test_data/test_smiles.smi', delimiter=' '))

    def test_get_mols_from_smilesfile(self):
        # Origin:
        # https://github.com/rinikerlab/lightweight-registration/blob/main/lwreg/test_lwreg.py
        filenames = [
            'test_data/test_smiles_no_delim.smi',
            'test_data/test_smiles_no_delim_with_header.smi',
            'test_data/test_smiles_with_header.smi',
            'test_data/test_smiles.smi'
        ]
        for filename in filenames:
            mols = _get_mols_from_smilesfile(filename)
            self.assertEqual(len(mols), 6)
