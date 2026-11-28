"""Role laws for variable-size native topology, separate from physical channel widths."""

from dataclasses import dataclass, replace

from r0.modules.priors.lib.categorical_converter_plans import CategoricalConverterPrior
from r0.modules.priors.lib.native_domain_prior import AsymmetricNativeDomainPrior, NativeDomainPrior

_CONTINUOUS = NativeDomainPrior(kinds=("continuous",))


@dataclass(frozen=True, slots=True)
class NativeTopologyRoles:
    roots: NativeDomainPrior = _CONTINUOUS
    hidden: NativeDomainPrior = _CONTINUOUS
    features: NativeDomainPrior = _CONTINUOUS
    target: NativeDomainPrior = _CONTINUOUS
    placement: str = "permuted-distinct-nodes-r0-v1"
    version: str = "native-topology-roles-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "native-topology-roles-r0-v1"
            or self.placement
            not in ("permuted-distinct-nodes-r0-v1", "prefix-features-last-target-r0-v1")
            or any(
                type(d) not in (NativeDomainPrior, AsymmetricNativeDomainPrior)
                or d.continuous_width != 1
                for d in (self.roots, self.hidden, self.features, self.target)
            )
        ):
            raise ValueError("native topology roles require scalar domain priors and placement law")

    def domain(self, *, root: bool, feature: bool, target: bool) -> NativeDomainPrior:
        # Readout role overrides structural root/hidden role explicitly.
        return (
            self.target
            if target
            else self.features
            if feature
            else self.roots
            if root
            else self.hidden
        )

    def maximum_cardinality(self) -> int:
        return max(
            max(d.cardinalities) if "categorical" in d.kinds else 2 if "binary" in d.kinds else 1
            for d in (self.roots, self.hidden, self.features, self.target)
        )

    def for_width(self, domain: NativeDomainPrior, width: int) -> NativeDomainPrior:
        return replace(domain, continuous_width=width)


@dataclass(frozen=True, slots=True)
class ConvertedTopologyRoles:
    vocabularies: tuple[tuple[str, ...], ...]
    target_vocabulary: tuple[str, ...] = ()
    converter: CategoricalConverterPrior = CategoricalConverterPrior()
    version: str = "converted-topology-roles-r0-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "converted-topology-roles-r0-v1"
            or type(self.vocabularies) is not tuple
            or not self.vocabularies
            or type(self.converter) is not CategoricalConverterPrior
        ):
            raise ValueError("converted topology roles require explicit feature vocabularies")
        for vocabulary in (*self.vocabularies, self.target_vocabulary):
            if (
                type(vocabulary) is not tuple
                or len(vocabulary) == 1
                or any(type(v) is not str or not v for v in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)
            ):
                raise ValueError(
                    "converted vocabulary requires zero or at least two distinct labels"
                )
