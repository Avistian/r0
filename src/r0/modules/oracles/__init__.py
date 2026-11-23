"""Predictive oracles conditioned only on declared assumptions and observations."""

from r0.modules.oracles.lib.classification import (
    ClassificationPrediction as ClassificationPrediction,
)
from r0.modules.oracles.lib.classification import ClassificationSpec as ClassificationSpec
from r0.modules.oracles.lib.classification import classification as classification
from r0.modules.oracles.lib.classification import classification_prior as classification_prior
from r0.modules.oracles.lib.gp import GPPrediction as GPPrediction
from r0.modules.oracles.lib.gp import GPSpec as GPSpec
from r0.modules.oracles.lib.gp import gp as gp
from r0.modules.oracles.lib.gp import gp_prior as gp_prior
