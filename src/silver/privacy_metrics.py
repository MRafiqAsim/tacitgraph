"""
Privacy Metrics for Anonymization Evaluation

Implements advanced privacy metrics:
- K-anonymity
- L-diversity
- T-closeness
- Re-identification Risk
- Closest Distances Ratio (CDR)

These metrics evaluate the quality of anonymization beyond simple
PII detection accuracy.
"""

import math
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class QuasiIdentifier:
    """
    A quasi-identifier attribute that could be used for re-identification.

    Examples: age range, zip code prefix, gender, job title
    """
    name: str
    value: Any
    generalization_level: int = 0  # 0 = original, higher = more generalized


@dataclass
class SensitiveAttribute:
    """
    A sensitive attribute that should be protected.

    Examples: salary, health condition, political affiliation
    """
    name: str
    value: Any


@dataclass
class AnonymizedRecord:
    """
    A record after anonymization with quasi-identifiers and sensitive attributes.
    """
    record_id: str
    quasi_identifiers: List[QuasiIdentifier]
    sensitive_attributes: List[SensitiveAttribute]
    original_text: Optional[str] = None
    anonymized_text: Optional[str] = None

    def get_qi_signature(self) -> str:
        """Get signature of quasi-identifiers for equivalence class grouping"""
        qi_values = tuple(sorted((qi.name, str(qi.value)) for qi in self.quasi_identifiers))
        return hashlib.md5(str(qi_values).encode()).hexdigest()


@dataclass
class EquivalenceClass:
    """
    A group of records with identical quasi-identifiers.
    """
    signature: str
    records: List[AnonymizedRecord] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.records)

    def get_sensitive_values(self, attribute_name: str) -> List[Any]:
        """Get all values of a sensitive attribute in this class"""
        values = []
        for record in self.records:
            for sa in record.sensitive_attributes:
                if sa.name == attribute_name:
                    values.append(sa.value)
        return values

    def get_distinct_sensitive_values(self, attribute_name: str) -> Set[Any]:
        """Get distinct values of a sensitive attribute"""
        return set(self.get_sensitive_values(attribute_name))


@dataclass
class PrivacyMetricsResult:
    """
    Complete privacy metrics evaluation result.
    """
    # K-anonymity
    k_anonymity: int
    k_anonymity_satisfied: bool
    k_threshold: int
    equivalence_class_sizes: List[int]

    # L-diversity
    l_diversity: int
    l_diversity_satisfied: bool
    l_threshold: int
    diversity_per_class: Dict[str, int]

    # T-closeness
    t_closeness: float
    t_closeness_satisfied: bool
    t_threshold: float
    distance_per_class: Dict[str, float]

    # Re-identification Risk
    reidentification_risk: float
    prosecutor_risk: float
    journalist_risk: float
    marketer_risk: float
    risk_threshold: float
    risk_satisfied: bool

    # Closest Distances Ratio
    cdr_score: float
    cdr_threshold: float
    cdr_satisfied: bool

    # Overall
    total_records: int
    total_equivalence_classes: int
    suppressed_records: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k_anonymity": {
                "k": self.k_anonymity,
                "threshold": self.k_threshold,
                "satisfied": self.k_anonymity_satisfied,
                "equivalence_classes": self.total_equivalence_classes,
                "class_sizes": {
                    "min": min(self.equivalence_class_sizes) if self.equivalence_class_sizes else 0,
                    "max": max(self.equivalence_class_sizes) if self.equivalence_class_sizes else 0,
                    "avg": sum(self.equivalence_class_sizes) / len(self.equivalence_class_sizes) if self.equivalence_class_sizes else 0,
                }
            },
            "l_diversity": {
                "l": self.l_diversity,
                "threshold": self.l_threshold,
                "satisfied": self.l_diversity_satisfied,
            },
            "t_closeness": {
                "t": self.t_closeness,
                "threshold": self.t_threshold,
                "satisfied": self.t_closeness_satisfied,
            },
            "reidentification_risk": {
                "overall": self.reidentification_risk,
                "prosecutor": self.prosecutor_risk,
                "journalist": self.journalist_risk,
                "marketer": self.marketer_risk,
                "threshold": self.risk_threshold,
                "satisfied": self.risk_satisfied,
            },
            "closest_distances_ratio": {
                "cdr": self.cdr_score,
                "threshold": self.cdr_threshold,
                "satisfied": self.cdr_satisfied,
            },
            "summary": {
                "total_records": self.total_records,
                "equivalence_classes": self.total_equivalence_classes,
                "suppressed_records": self.suppressed_records,
            }
        }

    def to_report(self) -> str:
        """Generate human-readable report"""
        k_status = "✓" if self.k_anonymity_satisfied else "✗"
        l_status = "✓" if self.l_diversity_satisfied else "✗"
        t_status = "✓" if self.t_closeness_satisfied else "✗"
        r_status = "✓" if self.risk_satisfied else "✗"
        c_status = "✓" if self.cdr_satisfied else "✗"

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║              PRIVACY METRICS EVALUATION REPORT                   ║
╠══════════════════════════════════════════════════════════════════╣

DATASET SUMMARY
───────────────
Total Records:           {self.total_records}
Equivalence Classes:     {self.total_equivalence_classes}
Suppressed Records:      {self.suppressed_records}

PRIVACY METRICS
───────────────
┌─────────────────────────┬──────────┬──────────┬────────┐
│ Metric                  │ Value    │ Threshold│ Status │
├─────────────────────────┼──────────┼──────────┼────────┤
│ K-Anonymity             │ k={self.k_anonymity:<6}│ k≥{self.k_threshold:<6}│ {k_status}      │
│ L-Diversity             │ l={self.l_diversity:<6}│ l≥{self.l_threshold:<6}│ {l_status}      │
│ T-Closeness             │ t={self.t_closeness:.4f}│ t≤{self.t_threshold:<6}│ {t_status}      │
│ Re-identification Risk  │ {self.reidentification_risk:.4f}  │ ≤{self.risk_threshold:<6}│ {r_status}      │
│ Closest Distances Ratio │ {self.cdr_score:.4f}  │ ≥{self.cdr_threshold:<6}│ {c_status}      │
└─────────────────────────┴──────────┴──────────┴────────┘

RE-IDENTIFICATION RISK BREAKDOWN
────────────────────────────────
Prosecutor Risk:  {self.prosecutor_risk:.4f} (targeted attack)
Journalist Risk:  {self.journalist_risk:.4f} (known target in dataset)
Marketer Risk:    {self.marketer_risk:.4f} (random record selection)

EQUIVALENCE CLASS DISTRIBUTION
──────────────────────────────
Smallest Class: {min(self.equivalence_class_sizes) if self.equivalence_class_sizes else 'N/A'} records
Largest Class:  {max(self.equivalence_class_sizes) if self.equivalence_class_sizes else 'N/A'} records
Average Size:   {sum(self.equivalence_class_sizes) / len(self.equivalence_class_sizes):.1f} records

"""
        return report


# =============================================================================
# Privacy Metrics Calculator
# =============================================================================

class PrivacyMetricsCalculator:
    """
    Calculate privacy metrics for anonymized datasets.

    Metrics:
    - K-anonymity: Each record is indistinguishable from k-1 others
    - L-diversity: Each equivalence class has l distinct sensitive values
    - T-closeness: Distribution of sensitive values is close to overall
    - Re-identification Risk: Probability of identifying an individual
    - Closest Distances Ratio: Measure of record distinguishability
    """

    def __init__(
        self,
        k_threshold: int = 5,
        l_threshold: int = 3,
        t_threshold: float = 0.2,
        risk_threshold: float = 0.1,
        cdr_threshold: float = 0.5
    ):
        """
        Initialize the calculator with thresholds.

        Args:
            k_threshold: Minimum k for k-anonymity (default: 5)
            l_threshold: Minimum l for l-diversity (default: 3)
            t_threshold: Maximum t for t-closeness (default: 0.2)
            risk_threshold: Maximum acceptable re-identification risk (default: 0.1)
            cdr_threshold: Minimum CDR score (default: 0.5)
        """
        self.k_threshold = k_threshold
        self.l_threshold = l_threshold
        self.t_threshold = t_threshold
        self.risk_threshold = risk_threshold
        self.cdr_threshold = cdr_threshold

    def calculate_all_metrics(
        self,
        records: List[AnonymizedRecord],
        sensitive_attribute: str = None
    ) -> PrivacyMetricsResult:
        """
        Calculate all privacy metrics for a dataset.

        Args:
            records: List of anonymized records
            sensitive_attribute: Name of sensitive attribute for L-diversity and T-closeness

        Returns:
            PrivacyMetricsResult with all metrics
        """
        if not records:
            return self._empty_result()

        # Build equivalence classes
        equivalence_classes = self._build_equivalence_classes(records)
        class_sizes = [ec.size for ec in equivalence_classes.values()]

        # K-anonymity
        k_anonymity = self._calculate_k_anonymity(equivalence_classes)

        # L-diversity
        l_diversity, diversity_per_class = self._calculate_l_diversity(
            equivalence_classes, sensitive_attribute
        )

        # T-closeness
        t_closeness, distance_per_class = self._calculate_t_closeness(
            equivalence_classes, records, sensitive_attribute
        )

        # Re-identification risks
        prosecutor, journalist, marketer = self._calculate_reidentification_risks(
            equivalence_classes, len(records)
        )
        overall_risk = max(prosecutor, journalist, marketer)

        # Closest Distances Ratio
        cdr_score = self._calculate_cdr(records)

        # Count suppressed records (records in very small classes)
        suppressed = sum(1 for ec in equivalence_classes.values() if ec.size < self.k_threshold)

        return PrivacyMetricsResult(
            # K-anonymity
            k_anonymity=k_anonymity,
            k_anonymity_satisfied=k_anonymity >= self.k_threshold,
            k_threshold=self.k_threshold,
            equivalence_class_sizes=class_sizes,

            # L-diversity
            l_diversity=l_diversity,
            l_diversity_satisfied=l_diversity >= self.l_threshold,
            l_threshold=self.l_threshold,
            diversity_per_class=diversity_per_class,

            # T-closeness
            t_closeness=t_closeness,
            t_closeness_satisfied=t_closeness <= self.t_threshold,
            t_threshold=self.t_threshold,
            distance_per_class=distance_per_class,

            # Re-identification Risk
            reidentification_risk=overall_risk,
            prosecutor_risk=prosecutor,
            journalist_risk=journalist,
            marketer_risk=marketer,
            risk_threshold=self.risk_threshold,
            risk_satisfied=overall_risk <= self.risk_threshold,

            # CDR
            cdr_score=cdr_score,
            cdr_threshold=self.cdr_threshold,
            cdr_satisfied=cdr_score >= self.cdr_threshold,

            # Summary
            total_records=len(records),
            total_equivalence_classes=len(equivalence_classes),
            suppressed_records=suppressed,
        )

    def _build_equivalence_classes(
        self,
        records: List[AnonymizedRecord]
    ) -> Dict[str, EquivalenceClass]:
        """Group records into equivalence classes by quasi-identifiers"""
        classes = {}

        for record in records:
            signature = record.get_qi_signature()

            if signature not in classes:
                classes[signature] = EquivalenceClass(signature=signature)

            classes[signature].records.append(record)

        return classes

    # =========================================================================
    # K-Anonymity
    # =========================================================================

    def _calculate_k_anonymity(
        self,
        equivalence_classes: Dict[str, EquivalenceClass]
    ) -> int:
        """
        Calculate k-anonymity level.

        K-anonymity = size of smallest equivalence class

        A dataset satisfies k-anonymity if every record is indistinguishable
        from at least k-1 other records based on quasi-identifiers.
        """
        if not equivalence_classes:
            return 0

        min_class_size = min(ec.size for ec in equivalence_classes.values())
        return min_class_size

    # =========================================================================
    # L-Diversity
    # =========================================================================

    def _calculate_l_diversity(
        self,
        equivalence_classes: Dict[str, EquivalenceClass],
        sensitive_attribute: Optional[str]
    ) -> Tuple[int, Dict[str, int]]:
        """
        Calculate l-diversity level.

        L-diversity = minimum number of distinct sensitive values
        in any equivalence class.

        A dataset satisfies l-diversity if every equivalence class
        contains at least l "well-represented" values for sensitive attributes.
        """
        if not equivalence_classes or not sensitive_attribute:
            return 0, {}

        diversity_per_class = {}
        min_diversity = float('inf')

        for signature, ec in equivalence_classes.items():
            distinct_values = ec.get_distinct_sensitive_values(sensitive_attribute)
            diversity = len(distinct_values)
            diversity_per_class[signature[:8]] = diversity
            min_diversity = min(min_diversity, diversity)

        if min_diversity == float('inf'):
            min_diversity = 0

        return int(min_diversity), diversity_per_class

    def calculate_entropy_l_diversity(
        self,
        equivalence_classes: Dict[str, EquivalenceClass],
        sensitive_attribute: str
    ) -> float:
        """
        Calculate entropy l-diversity.

        More robust than simple l-diversity as it considers
        the distribution of sensitive values.
        """
        if not equivalence_classes or not sensitive_attribute:
            return 0.0

        min_entropy = float('inf')

        for ec in equivalence_classes.values():
            values = ec.get_sensitive_values(sensitive_attribute)
            if not values:
                continue

            # Calculate entropy
            value_counts = Counter(values)
            total = len(values)
            entropy = 0.0

            for count in value_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            min_entropy = min(min_entropy, entropy)

        if min_entropy == float('inf'):
            return 0.0

        # Convert entropy to equivalent l value
        return 2 ** min_entropy

    # =========================================================================
    # T-Closeness
    # =========================================================================

    def _calculate_t_closeness(
        self,
        equivalence_classes: Dict[str, EquivalenceClass],
        all_records: List[AnonymizedRecord],
        sensitive_attribute: Optional[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate t-closeness.

        T-closeness measures how close the distribution of sensitive
        attributes in each equivalence class is to the overall distribution.

        Uses Earth Mover's Distance (EMD) for numerical attributes
        and Variation Distance for categorical attributes.
        """
        if not equivalence_classes or not sensitive_attribute or not all_records:
            return 0.0, {}

        # Get overall distribution
        all_values = []
        for record in all_records:
            for sa in record.sensitive_attributes:
                if sa.name == sensitive_attribute:
                    all_values.append(sa.value)

        if not all_values:
            return 0.0, {}

        overall_dist = self._get_distribution(all_values)

        distance_per_class = {}
        max_distance = 0.0

        for signature, ec in equivalence_classes.items():
            class_values = ec.get_sensitive_values(sensitive_attribute)

            if not class_values:
                continue

            class_dist = self._get_distribution(class_values)

            # Calculate distance (using variation distance for categorical)
            distance = self._variation_distance(overall_dist, class_dist)
            distance_per_class[signature[:8]] = distance
            max_distance = max(max_distance, distance)

        return max_distance, distance_per_class

    def _get_distribution(self, values: List[Any]) -> Dict[Any, float]:
        """Get probability distribution of values"""
        counts = Counter(values)
        total = len(values)
        return {v: c / total for v, c in counts.items()}

    def _variation_distance(
        self,
        dist1: Dict[Any, float],
        dist2: Dict[Any, float]
    ) -> float:
        """
        Calculate variation distance between two distributions.

        Variation distance = 0.5 * sum(|P(x) - Q(x)|) for all x
        """
        all_keys = set(dist1.keys()) | set(dist2.keys())

        distance = 0.0
        for key in all_keys:
            p1 = dist1.get(key, 0.0)
            p2 = dist2.get(key, 0.0)
            distance += abs(p1 - p2)

        return distance / 2.0

    def _earth_movers_distance(
        self,
        dist1: Dict[float, float],
        dist2: Dict[float, float]
    ) -> float:
        """
        Calculate Earth Mover's Distance for numerical attributes.

        EMD measures the minimum "work" needed to transform one
        distribution into another.
        """
        # Sort by value for numerical attributes
        values1 = sorted(dist1.keys())
        values2 = sorted(dist2.keys())
        all_values = sorted(set(values1) | set(values2))

        # Calculate cumulative distributions
        cdf1 = {}
        cdf2 = {}
        cum1, cum2 = 0.0, 0.0

        for v in all_values:
            cum1 += dist1.get(v, 0.0)
            cum2 += dist2.get(v, 0.0)
            cdf1[v] = cum1
            cdf2[v] = cum2

        # EMD = integral of |CDF1 - CDF2|
        emd = 0.0
        prev_v = all_values[0]

        for v in all_values[1:]:
            diff = abs(cdf1.get(prev_v, 0) - cdf2.get(prev_v, 0))
            emd += diff * (v - prev_v)
            prev_v = v

        # Normalize by range
        value_range = all_values[-1] - all_values[0]
        if value_range > 0:
            emd /= value_range

        return emd

    # =========================================================================
    # Re-identification Risk
    # =========================================================================

    def _calculate_reidentification_risks(
        self,
        equivalence_classes: Dict[str, EquivalenceClass],
        total_records: int
    ) -> Tuple[float, float, float]:
        """
        Calculate re-identification risk metrics.

        Returns:
            (prosecutor_risk, journalist_risk, marketer_risk)

        Prosecutor Risk: Attacker knows target is in dataset
        Journalist Risk: Attacker knows target is in dataset AND can verify match
        Marketer Risk: Attacker randomly selects records
        """
        if not equivalence_classes or total_records == 0:
            return 1.0, 1.0, 1.0

        # Prosecutor risk = max(1/|EC|) for all equivalence classes
        # This is the worst case: attacker targets someone in smallest class
        prosecutor_risk = max(1.0 / ec.size for ec in equivalence_classes.values())

        # Journalist risk = sum of (1/|EC|) for records in classes where |EC| = 1
        # Journalist can verify if a match is correct
        unique_records = sum(1 for ec in equivalence_classes.values() if ec.size == 1)
        journalist_risk = unique_records / total_records

        # Marketer risk = 1/n for random record selection
        # Expected probability of re-identifying a random record
        marketer_risk = sum(
            (ec.size / total_records) * (1.0 / ec.size)
            for ec in equivalence_classes.values()
        )

        return prosecutor_risk, journalist_risk, marketer_risk

    def calculate_individual_risk(
        self,
        record: AnonymizedRecord,
        equivalence_classes: Dict[str, EquivalenceClass]
    ) -> float:
        """
        Calculate re-identification risk for an individual record.

        Risk = 1 / size of record's equivalence class
        """
        signature = record.get_qi_signature()

        if signature in equivalence_classes:
            return 1.0 / equivalence_classes[signature].size

        return 1.0  # Record is unique

    # =========================================================================
    # Closest Distances Ratio (CDR)
    # =========================================================================

    def _calculate_cdr(
        self,
        records: List[AnonymizedRecord],
        sample_size: int = 100
    ) -> float:
        """
        Calculate Closest Distances Ratio.

        CDR measures how well anonymization preserves distances between records.

        CDR = avg(distance to closest different-class record) /
              avg(distance to closest same-class record)

        Higher CDR means better separation between equivalence classes.
        """
        if len(records) < 2:
            return 1.0

        # Sample if dataset is large
        if len(records) > sample_size:
            import random
            sampled_records = random.sample(records, sample_size)
        else:
            sampled_records = records

        same_class_distances = []
        diff_class_distances = []

        for i, record1 in enumerate(sampled_records):
            sig1 = record1.get_qi_signature()

            min_same = float('inf')
            min_diff = float('inf')

            for j, record2 in enumerate(sampled_records):
                if i == j:
                    continue

                sig2 = record2.get_qi_signature()
                distance = self._record_distance(record1, record2)

                if sig1 == sig2:
                    min_same = min(min_same, distance)
                else:
                    min_diff = min(min_diff, distance)

            if min_same < float('inf'):
                same_class_distances.append(min_same)
            if min_diff < float('inf'):
                diff_class_distances.append(min_diff)

        # Calculate CDR
        if not same_class_distances or not diff_class_distances:
            return 1.0

        avg_same = sum(same_class_distances) / len(same_class_distances)
        avg_diff = sum(diff_class_distances) / len(diff_class_distances)

        if avg_same == 0:
            return float('inf') if avg_diff > 0 else 1.0

        return avg_diff / avg_same

    def _record_distance(
        self,
        record1: AnonymizedRecord,
        record2: AnonymizedRecord
    ) -> float:
        """
        Calculate distance between two records based on quasi-identifiers.

        Uses normalized Hamming distance for categorical QIs
        and normalized Euclidean distance for numerical QIs.
        """
        if not record1.quasi_identifiers or not record2.quasi_identifiers:
            return 0.0

        # Create QI dictionaries
        qi1 = {qi.name: qi.value for qi in record1.quasi_identifiers}
        qi2 = {qi.name: qi.value for qi in record2.quasi_identifiers}

        common_qis = set(qi1.keys()) & set(qi2.keys())

        if not common_qis:
            return 0.0

        distance = 0.0
        for qi_name in common_qis:
            v1, v2 = qi1[qi_name], qi2[qi_name]

            # Try numerical distance
            try:
                num1, num2 = float(v1), float(v2)
                # Normalized distance (assuming values are already normalized)
                distance += abs(num1 - num2)
            except (ValueError, TypeError):
                # Categorical: 0 if same, 1 if different
                distance += 0 if v1 == v2 else 1

        return distance / len(common_qis)

    def _empty_result(self) -> PrivacyMetricsResult:
        """Return empty result for empty dataset"""
        return PrivacyMetricsResult(
            k_anonymity=0,
            k_anonymity_satisfied=False,
            k_threshold=self.k_threshold,
            equivalence_class_sizes=[],
            l_diversity=0,
            l_diversity_satisfied=False,
            l_threshold=self.l_threshold,
            diversity_per_class={},
            t_closeness=1.0,
            t_closeness_satisfied=False,
            t_threshold=self.t_threshold,
            distance_per_class={},
            reidentification_risk=1.0,
            prosecutor_risk=1.0,
            journalist_risk=1.0,
            marketer_risk=1.0,
            risk_threshold=self.risk_threshold,
            risk_satisfied=False,
            cdr_score=0.0,
            cdr_threshold=self.cdr_threshold,
            cdr_satisfied=False,
            total_records=0,
            total_equivalence_classes=0,
            suppressed_records=0,
        )


# =============================================================================
# Text-based Privacy Metrics
# =============================================================================

class TextPrivacyAnalyzer:
    """
    Analyze privacy metrics for text anonymization.

    Converts anonymized text documents into structured records
    for privacy metric calculation.
    """

    def __init__(self, calculator: Optional[PrivacyMetricsCalculator] = None):
        self.calculator = calculator or PrivacyMetricsCalculator()

    def analyze_anonymized_texts(
        self,
        original_texts: List[str],
        anonymized_texts: List[str],
        quasi_identifier_extractors: Dict[str, callable] = None,
        sensitive_attribute_extractor: callable = None
    ) -> PrivacyMetricsResult:
        """
        Analyze privacy metrics for anonymized texts.

        Args:
            original_texts: Original text documents
            anonymized_texts: Anonymized versions
            quasi_identifier_extractors: Functions to extract QIs from text
            sensitive_attribute_extractor: Function to extract sensitive attrs

        Returns:
            PrivacyMetricsResult
        """
        records = []

        for i, (original, anonymized) in enumerate(zip(original_texts, anonymized_texts)):
            # Extract quasi-identifiers from anonymized text
            qis = self._extract_quasi_identifiers(
                anonymized,
                quasi_identifier_extractors
            )

            # Extract sensitive attributes from original (for L-diversity check)
            sensitive_attrs = self._extract_sensitive_attributes(
                original,
                sensitive_attribute_extractor
            )

            record = AnonymizedRecord(
                record_id=f"doc_{i}",
                quasi_identifiers=qis,
                sensitive_attributes=sensitive_attrs,
                original_text=original,
                anonymized_text=anonymized
            )
            records.append(record)

        # Calculate metrics
        sensitive_attr_name = "content" if sensitive_attrs else None
        return self.calculator.calculate_all_metrics(records, sensitive_attr_name)

    def _extract_quasi_identifiers(
        self,
        text: str,
        extractors: Dict[str, callable] = None
    ) -> List[QuasiIdentifier]:
        """Extract quasi-identifiers from text"""
        qis = []

        if extractors:
            for name, extractor in extractors.items():
                try:
                    value = extractor(text)
                    if value is not None:
                        qis.append(QuasiIdentifier(name=name, value=value))
                except Exception:
                    pass
        else:
            # Default: use placeholder patterns as QIs
            import re

            # Count placeholder types as QI
            person_count = len(re.findall(r'\[PERSON_\d+\]', text))
            email_count = len(re.findall(r'\[EMAIL_\d+\]', text))
            phone_count = len(re.findall(r'\[PHONE_\d+\]', text))

            qis.append(QuasiIdentifier(name="person_count", value=person_count))
            qis.append(QuasiIdentifier(name="email_count", value=email_count))
            qis.append(QuasiIdentifier(name="phone_count", value=phone_count))

            # Text length bucket
            length_bucket = len(text) // 100 * 100
            qis.append(QuasiIdentifier(name="length_bucket", value=length_bucket))

        return qis

    def _extract_sensitive_attributes(
        self,
        text: str,
        extractor: callable = None
    ) -> List[SensitiveAttribute]:
        """Extract sensitive attributes from text"""
        if extractor:
            try:
                value = extractor(text)
                if value is not None:
                    return [SensitiveAttribute(name="content", value=value)]
            except Exception:
                pass

        # Default: hash of content as sensitive attribute
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return [SensitiveAttribute(name="content", value=content_hash)]


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_privacy_metrics(
    records: List[AnonymizedRecord],
    sensitive_attribute: str = None,
    k_threshold: int = 5,
    l_threshold: int = 3,
    t_threshold: float = 0.2
) -> PrivacyMetricsResult:
    """
    Calculate privacy metrics for anonymized records.

    Args:
        records: List of AnonymizedRecord objects
        sensitive_attribute: Name of sensitive attribute
        k_threshold: Minimum k for k-anonymity
        l_threshold: Minimum l for l-diversity
        t_threshold: Maximum t for t-closeness

    Returns:
        PrivacyMetricsResult with all metrics
    """
    calculator = PrivacyMetricsCalculator(
        k_threshold=k_threshold,
        l_threshold=l_threshold,
        t_threshold=t_threshold
    )
    return calculator.calculate_all_metrics(records, sensitive_attribute)


def analyze_text_privacy(
    original_texts: List[str],
    anonymized_texts: List[str]
) -> PrivacyMetricsResult:
    """
    Analyze privacy metrics for anonymized texts.

    Args:
        original_texts: Original documents
        anonymized_texts: Anonymized documents

    Returns:
        PrivacyMetricsResult
    """
    analyzer = TextPrivacyAnalyzer()
    return analyzer.analyze_anonymized_texts(original_texts, anonymized_texts)
