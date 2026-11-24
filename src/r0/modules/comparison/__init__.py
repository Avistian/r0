"""Offline fixture comparison with fixed rows, isolated inputs and immutable evidence."""

from .lib.adapters import Predictor as Predictor
from .lib.adapters import predict_adapter as predict_adapter
from .lib.execution import environment as environment
from .lib.execution import run_process as run_process
from .lib.predictors import Unsupported as Unsupported
from .lib.predictors import predict as predict
from .lib.predictors import recipe as recipe
from .lib.prepare import prepare as prepare
from .lib.prepare import row_identity as row_identity
from .lib.runner import attempt as attempt
from .lib.runner import cache_identity as cache_identity
from .lib.runner import compare as compare
from .lib.runner import read_attempt as read_attempt
from .lib.scoring import check_invariance as check_invariance
from .lib.scoring import score as score
from .lib.summary import aggregate_datasets as aggregate_datasets
from .lib.summary import comparison_gaps as comparison_gaps
from .lib.tabpfn import tabpfn_status as tabpfn_status
