import numpy as np

import quotonic.fock as fock
import quotonic.utils as utils


def test_comp_to_secq():
    result = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    assert np.allclose(utils.comp_to_secq(np.array([1, 0, 0, 1])), result)


def test_symm_fock_to_comp():
    result = np.array([1, 0, 0, 1])
    assert np.allclose(utils.secq_to_comp(np.array([0, 1, 1, 0, 1, 0, 0, 1])), result)


def test_comp_indices_from_symm_fock():
    n = 4
    m = 8
    basis = fock.build_secq_basis(n, m)
    result = np.array([77, 78, 80, 81, 92, 93, 95, 96, 161, 162, 164, 165, 176, 177, 179, 180])
    assert np.allclose(utils.comp_indices_from_secq(basis), result)

def test_genGGMSet():
    ggm_set = utils.genGGMSet(m=3)
    assert ggm_set.shape == (8, 3, 3), f"Expected shape (8, 3, 3), got {ggm_set.shape}"
    expected_first = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    assert np.allclose(ggm_set[0], expected_first), "First GGM matrix does not match expected value."
    expected_second = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    assert np.allclose(ggm_set[2], expected_second), "Third GGM matrix does not match expected value."
    expected_last = np.array([[1/np.sqrt(3), 0, 0], [0, 1/np.sqrt(3), 0], [0, 0, -2/np.sqrt(3)]], dtype=complex)
    assert np.allclose(ggm_set[7], expected_last), "Eighth GGM matrix does not match expected value."

def test_genWeylSet():
    weyl_set = utils.genWeylSet(m=3, vectorized=True)
    assert weyl_set.shape == (9, 3, 3), f"Expected shape (9, 3, 3), got {weyl_set.shape}"
    expected_first = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=complex)
    assert np.allclose(weyl_set[0], expected_first), "First Weyl matrix does not match expected value."
    expected_second = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
    assert np.allclose(weyl_set[1], expected_second), "Second Weyl matrix does not match expected value."
    expected_last = np.array([[0, 0, 1], [np.exp(-2*np.pi*1j/3), 0, 0], [0, np.exp(2*np.pi*1j/3), 0]], dtype=complex)
    assert np.allclose(weyl_set[8], expected_last), "Third Weyl matrix does not match expected value."
