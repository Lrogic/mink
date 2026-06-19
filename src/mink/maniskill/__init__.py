"""ManiSkill/SAPIEN kinematics backend for mink.

This subpackage provides a drop-in :class:`Configuration` equivalent backed by
``pytorch_kinematics`` (a ManiSkill dependency) parsing a URDF, plus the few
tasks/limits that need a backend-specific reimplementation. It is imported
separately from the top-level ``mink`` package so that ``import mink`` never
pulls in SAPIEN/torch.
"""

from .configuration import ManiSkillConfiguration as ManiSkillConfiguration
from .tasks import ManiSkillConfigurationLimit as ManiSkillConfigurationLimit
from .tasks import ManiSkillPostureTask as ManiSkillPostureTask
