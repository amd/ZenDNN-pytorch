from .backends import BACKENDS
from .training import register_aot_and_inductor_training_backends

register_aot_and_inductor_training_backends()

__all__ = ["BACKENDS"]
