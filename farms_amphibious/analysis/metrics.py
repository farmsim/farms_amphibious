"""Metrics"""

from farms_core.analysis.metrics import analyse_gait
from ..model.convention import AmphibiousConvention


def analyse_gait_amphibious(animat_data, animat_options):
    """Analyse gait"""
    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    contact_indices = range(convention.n_legs)
    joint_indices = [
        convention.legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0)
        for leg_i in range(animat_options.morphology.n_legs//2)
        for side_i in range(2)
    ]
    return analyse_gait(
        animat_data=animat_data,
        animat_options=animat_options,
        contact_indices=contact_indices,
        joint_indices=joint_indices,
    )
