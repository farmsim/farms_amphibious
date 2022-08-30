"""Ekeberg muscle model"""

cimport numpy as np
import numpy as np
from .muscle_cy cimport JointsMusclesCy


cdef class EkebergMuscleCy(JointsMusclesCy):
    """Ekeberg muscle model"""
    cdef public np.ndarray joints_offsets
    cpdef void step(self, unsigned int iteration)
