"""Ekeberg muscle model"""

include 'types.pxd'
cimport numpy as np
import numpy as np
from .muscle_cy cimport JointsMusclesCy


cdef class EkebergMuscleCy(JointsMusclesCy):
    """Ekeberg muscle model"""
    cdef public np.ndarray joints_offsets
    cdef public np.ndarray activations
    cpdef void update_activations(self, unsigned int iteration)
    cpdef void update_offsets(self, unsigned int iteration)
    cpdef void step(self, unsigned int iteration)
