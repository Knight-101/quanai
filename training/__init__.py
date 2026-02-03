"""
Training Module

This module provides advanced training algorithms and curriculum
learning for the trading system.
"""

from .curriculum import CurriculumScheduler, TrainingManager

__all__ = ['CurriculumScheduler', 'TrainingManager'] 