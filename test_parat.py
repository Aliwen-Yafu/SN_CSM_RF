# test_parat.py
from Parat import Parameters, Spectrum

def test_parameters_init():
    params = Parameters(1e51, 16, 1.34e-4, 12, 0.1, 2, 0.01, 6e13)
    assert params.kineticEnergy == 1e51

def test_spectrum_cr():
    spec = Spectrum(2, 1, 10)
    result = spec.CR(1)
    assert result > 0
