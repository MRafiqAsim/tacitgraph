"""
Anonymization Evaluator

Evaluates PII detection quality across processing modes:
- Precision/Recall/F1 per PII type
- Identity consistency score (same person → same pseudonym)
- False positive analysis by error category
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

logger = logging.getLogger(__name__)


@dataclass
class PIIAnnotation:
    """A ground-truth PII annotation"""
    text: str
    pii_type: str
    start: int
    end: int


@dataclass
class PIITypeMetrics:
    """Precision/Recall/F1 for a single PII type"""
    pii_type: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pii_type": self.pii_type,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class IdentityConsistencyResult:
    """Results of identity consistency evaluation"""
    total_person_entities: int = 0
    unique_pseudonyms: int = 0
    consistency_score: float = 0.0  # 1.0 = perfect (all refs to same person → same ID)
    inconsistencies: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_person_entities": self.total_person_entities,
            "unique_pseudonyms": self.unique_pseudonyms,
            "consistency_score": round(self.consistency_score, 4),
            "inconsistency_count": len(self.inconsistencies),
            "inconsistencies": self.inconsistencies[:20],  # Top 20
        }


@dataclass
class FalsePositiveAnalysis:
    """Analysis of false positive errors by category"""
    categories: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            category: {"count": len(items), "examples": items[:5]}
            for category, items in self.categories.items()
        }


class AnonymizationEvaluator:
    """
    Evaluate PII detection and anonymization quality.

    Compares detected PII entities against ground truth annotations
    and evaluates identity consistency across the corpus.
    """

    def __init__(self, ground_truth_path: Optional[str] = None):
        """
        Args:
            ground_truth_path: Path to ground_truth_pii.json with manual annotations
        """
        self.ground_truth: Dict[str, List[PIIAnnotation]] = {}

        if ground_truth_path and Path(ground_truth_path).exists():
            self._load_ground_truth(ground_truth_path)

    def _load_ground_truth(self, path: str) -> None:
        """Load ground truth annotations from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            email_id = item.get("email_id", item.get("chunk_id", ""))
            annotations = []
            for ann in item.get("annotations", []):
                annotations.append(PIIAnnotation(
                    text=ann["text"],
                    pii_type=ann["type"],
                    start=ann.get("start", 0),
                    end=ann.get("end", 0),
                ))
            self.ground_truth[email_id] = annotations

        logger.info(f"Loaded ground truth for {len(self.ground_truth)} documents")

    def evaluate_detection(
        self,
        silver_path: str,
        overlap_threshold: float = 0.5,
    ) -> Dict[str, PIITypeMetrics]:
        """
        Evaluate PII detection against ground truth.

        Uses span overlap to match predicted entities with ground truth.

        Args:
            silver_path: Path to silver layer with processed chunks
            overlap_threshold: Minimum overlap ratio to count as match

        Returns:
            Dict of PII type → metrics
        """
        if not self.ground_truth:
            logger.warning("No ground truth loaded — cannot evaluate detection")
            return {}

        metrics: Dict[str, PIITypeMetrics] = defaultdict(lambda: PIITypeMetrics(pii_type=""))

        silver = Path(silver_path)
        for chunk_dir in ["not_personal/email_chunks"]:
            chunk_path = silver / chunk_dir
            if not chunk_path.exists():
                continue

            for json_file in chunk_path.rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        chunk = json.load(f)

                    chunk_id = chunk.get("chunk_id", "")
                    if chunk_id not in self.ground_truth:
                        continue

                    gt_entities = self.ground_truth[chunk_id]
                    pred_entities = chunk.get("pii_entities", [])

                    self._match_entities(gt_entities, pred_entities, metrics, overlap_threshold)

                except Exception as e:
                    logger.debug(f"Error evaluating {json_file}: {e}")

        # Fix pii_type field
        for key in metrics:
            metrics[key].pii_type = key

        return dict(metrics)

    def _match_entities(
        self,
        ground_truth: List[PIIAnnotation],
        predictions: List[Dict[str, Any]],
        metrics: Dict[str, PIITypeMetrics],
        overlap_threshold: float,
    ) -> None:
        """Match predicted entities against ground truth using span overlap."""
        matched_gt = set()
        matched_pred = set()

        for i, gt in enumerate(ground_truth):
            for j, pred in enumerate(predictions):
                if j in matched_pred:
                    continue

                # Check type match
                if gt.pii_type != pred.get("type", ""):
                    continue

                # Check span overlap
                overlap = self._span_overlap(
                    gt.start, gt.end,
                    pred.get("start", 0), pred.get("end", 0)
                )

                gt_len = max(gt.end - gt.start, 1)
                if overlap / gt_len >= overlap_threshold:
                    metrics[gt.pii_type].true_positives += 1
                    matched_gt.add(i)
                    matched_pred.add(j)
                    break

        # False negatives: ground truth not matched
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt:
                metrics[gt.pii_type].false_negatives += 1

        # False positives: predictions not matched
        for j, pred in enumerate(predictions):
            if j not in matched_pred:
                pii_type = pred.get("type", "UNKNOWN")
                metrics[pii_type].false_positives += 1

    @staticmethod
    def _span_overlap(s1: int, e1: int, s2: int, e2: int) -> int:
        """Calculate character overlap between two spans."""
        return max(0, min(e1, e2) - max(s1, s2))

    def evaluate_identity_consistency(self, silver_path: str) -> IdentityConsistencyResult:
        """
        Evaluate whether the same person always gets the same pseudonym.

        Checks all PERSON entities across all chunks and verifies that
        identical or equivalent original texts map to the same replacement.

        Args:
            silver_path: Path to silver layer

        Returns:
            IdentityConsistencyResult
        """
        # Map: original person text → set of pseudonyms assigned
        person_pseudonyms: Dict[str, Set[str]] = defaultdict(set)
        total_person = 0

        silver = Path(silver_path)
        for chunk_dir in ["not_personal/email_chunks"]:
            chunk_path = silver / chunk_dir
            if not chunk_path.exists():
                continue

            for json_file in chunk_path.rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        chunk = json.load(f)

                    original = chunk.get("text_original", "")
                    anonymized = chunk.get("text_anonymized", "")

                    for entity in chunk.get("pii_entities", []):
                        if entity.get("type") != "PERSON":
                            continue

                        total_person += 1
                        person_text = entity.get("text", "")
                        start = entity.get("start", 0)
                        end = entity.get("end", 0)

                        # Find what it was replaced with in the anonymized text
                        # This is approximate — look for [PERSON_*] pattern near the position
                        pseudonym_match = re.search(
                            r'\[PERSON_\d+\]',
                            anonymized[max(0, start - 20):end + 20]
                        )
                        if pseudonym_match:
                            person_pseudonyms[person_text.lower()].add(pseudonym_match.group())

                except Exception as e:
                    logger.debug(f"Error processing {json_file}: {e}")

        # Calculate consistency
        inconsistencies = []
        consistent = 0
        total = 0

        for person, pseudonyms in person_pseudonyms.items():
            total += 1
            if len(pseudonyms) == 1:
                consistent += 1
            else:
                inconsistencies.append({
                    "person": person,
                    "pseudonyms": sorted(pseudonyms),
                    "count": len(pseudonyms),
                })

        consistency_score = consistent / total if total > 0 else 1.0

        return IdentityConsistencyResult(
            total_person_entities=total_person,
            unique_pseudonyms=len(set(p for ps in person_pseudonyms.values() for p in ps)),
            consistency_score=consistency_score,
            inconsistencies=sorted(inconsistencies, key=lambda x: -x["count"]),
        )

    def analyze_false_positives(self, silver_path: str) -> FalsePositiveAnalysis:
        """
        Categorize false positive detections.

        Categories:
        - dates_as_phones: Date patterns detected as PHONE
        - ips_as_phones: IP addresses detected as PHONE
        - common_words_as_persons: Common words detected as PERSON
        - header_labels_as_persons: Email header labels detected as PERSON
        - short_strings_as_persons: Very short strings detected as PERSON
        """
        categories: Dict[str, List[str]] = defaultdict(list)

        silver = Path(silver_path)
        for chunk_dir in ["not_personal/email_chunks"]:
            chunk_path = silver / chunk_dir
            if not chunk_path.exists():
                continue

            for json_file in chunk_path.rglob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        chunk = json.load(f)

                    for entity in chunk.get("pii_entities", []):
                        text = entity.get("text", "")
                        pii_type = entity.get("type", "")

                        if pii_type == "PHONE":
                            if re.match(r'\d{4}[-/]\d{2}[-/]\d{2}', text):
                                categories["dates_as_phones"].append(text)
                            elif re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text):
                                categories["ips_as_phones"].append(text)

                        elif pii_type == "PERSON":
                            if len(text) < 3:
                                categories["short_strings_as_persons"].append(text)
                            elif text.lower() in {
                                "tel", "fax", "cell", "sent", "received",
                                "from", "to", "cc", "subject", "date",
                                "regards", "thanks", "goodbye", "se",
                            }:
                                categories["common_words_as_persons"].append(text)
                            elif re.match(r'^(From|To|Cc|Date|Subject|Sent):', text):
                                categories["header_labels_as_persons"].append(text)

                except Exception:
                    continue

        return FalsePositiveAnalysis(categories=dict(categories))

    def generate_report(
        self,
        silver_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive anonymization evaluation report.

        Args:
            silver_path: Path to silver layer to evaluate
            output_path: Optional path to save report JSON

        Returns:
            Complete evaluation report as dict
        """
        report = {
            "silver_path": silver_path,
        }

        # Detection metrics (only if ground truth available)
        if self.ground_truth:
            detection_metrics = self.evaluate_detection(silver_path)
            report["detection_metrics"] = {
                k: v.to_dict() for k, v in detection_metrics.items()
            }

            # Aggregate metrics
            total_tp = sum(m.true_positives for m in detection_metrics.values())
            total_fp = sum(m.false_positives for m in detection_metrics.values())
            total_fn = sum(m.false_negatives for m in detection_metrics.values())
            agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0
            report["aggregate_detection"] = {
                "precision": round(agg_p, 4),
                "recall": round(agg_r, 4),
                "f1": round(agg_f1, 4),
            }

        # Identity consistency
        consistency = self.evaluate_identity_consistency(silver_path)
        report["identity_consistency"] = consistency.to_dict()

        # False positive analysis
        fp_analysis = self.analyze_false_positives(silver_path)
        report["false_positive_analysis"] = fp_analysis.to_dict()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation report saved to {output_path}")

        return report
