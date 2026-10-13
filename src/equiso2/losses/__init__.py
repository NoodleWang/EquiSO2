"""Structure-balanced squared fitting objectives."""
from .energy_force import energy_loss
from .objectives import force_loss
from .hessian import full_hessian_loss, hessian_action_loss, reference_hessian_actions

# Public terminology. Legacy internal names remain for compatibility.
hessian_vector_product_loss = hessian_action_loss
reference_hessian_vector_products = reference_hessian_actions
__all__ = ["energy_loss", "force_loss", "full_hessian_loss", "hessian_vector_product_loss", "reference_hessian_vector_products"]
