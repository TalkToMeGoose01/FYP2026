#!/usr/bin/env python
"""
check_versions.py
==================
Run this FIRST, before rerun_tft_confirmation.py, inside the same WSL2
environment / conda env you used for the original Phase 2 TFT runs.

Just prints the interpreter, TF, and GPU info so you can eyeball it
against the recorded environment (Python 3.11.15, TF 2.20.0, RTX 4070
SUPER, per archive/2026-08-17_analysis6/PROVENANCE.txt and
requirements_frozen.txt) before spending GPU time on the real run.
"""
import sys
print("=" * 60)
print("sys.executable :", sys.executable)
print("Python version :", sys.version)
print("=" * 60)

import tensorflow as tf
print("TF version     :", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print("GPUs visible   :", gpus)
for g in gpus:
    try:
        details = tf.config.experimental.get_device_details(g)
        print("  ->", details)
    except Exception as e:
        print("  -> (could not get device details:", e, ")")

try:
    import shap
    print("SHAP version   :", shap.__version__)
except ImportError:
    print("SHAP version   : not installed (not needed for this rerun)")

print("=" * 60)
print("Recorded original environment (for comparison):")
print("  Python 3.11.15, TensorFlow 2.20.0, SHAP 0.51.0, RTX 4070 SUPER (9,513MB)")
print("  Source: archive/2026-08-17_analysis6/PROVENANCE.txt")
print("=" * 60)
