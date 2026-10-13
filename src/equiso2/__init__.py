"""EquiSO2: conservative potentials and curvature-prioritized learning."""
from .models.hess_so2_equiformer import HessSO2Config, HessSO2Equiformer
from .api import paper_config, make_model, energy_force_hessian

__version__ = "0.1.0"
__all__ = ["HessSO2Config", "HessSO2Equiformer", "paper_config", "make_model", "energy_force_hessian"]
