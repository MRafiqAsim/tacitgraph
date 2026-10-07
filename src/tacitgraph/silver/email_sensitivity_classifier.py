"""
Email Sensitivity Classifier

Classifies emails as 'personal' or 'not_personal' using a weighted scoring
model across five signals:
  1. Subject line patterns  (0.30)
  2. Body content patterns  (0.30)
  3. Folder path context    (0.15)
  4. Recipient patterns     (0.15)
  5. Attachment types       (0.10)

Thread-level rule:
  If ANY email in a thread is classified as 'personal',
  the entire thread is skipped (saved to personal/ folder).

  Default (no classification or 'not_personal') → process normally.
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from tacitgraph.paths import CONFIG_DIR

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of email sensitivity classification with audit trail."""

    classification: str  # "personal" | "not_personal"
    confidence: float  # 0.0–1.0
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensitivityResult":
        return cls(
            classification=data.get("classification", "not_personal"),
            confidence=data.get("confidence", 0.5),
            signals=data.get("signals", {}),
        )


# Default pattern lists — can be overridden via sensitivity_rules.yaml
DEFAULT_TECHNICAL_SUBJECT = [
    # Software / IT
    r"(?i)\b(deployment|release|hotfix|patch|rollback)\b",
    r"(?i)\b(pipeline|CI/?CD|build|jenkins|github\s*actions?)\b",
    r"(?i)\b(API|endpoint|microservice|service\s*mesh)\b",
    r"(?i)\b(migration|schema\s*change|database|SQL)\b",
    r"(?i)\b(architecture|design\s*review|RFC|ADR)\b",
    r"(?i)\b(incident|outage|postmortem|root\s*cause)\b",
    r"(?i)\b(sprint|standup|retro|backlog|jira|ticket)\b",
    r"(?i)\b(code\s*review|pull\s*request|merge|branch)\b",
    r"(?i)\b(docker|kubernetes|k8s|terraform|ansible)\b",
    r"(?i)\b(monitoring|alerting|grafana|prometheus|datadog)\b",
    r"(?i)\b(config|configuration|environment|staging|production)\b",
    r"(?i)\b(ETL|data\s*pipeline|ingestion|transformation)\b",
    # Manufacturing / production
    r"(?i)\b(production\s*(schedule|report|line|order|batch))\b",
    r"(?i)\b(quality\s*(control|assurance|report|inspection|check))\b",
    r"(?i)\b(batch\s*(record|number|report|processing))\b",
    r"(?i)\b(equipment\s*(maintenance|failure|calibration|downtime))\b",
    r"(?i)\b(yield|throughput|cycle\s*time|downtime|OEE)\b",
    r"(?i)\b(BOM|bill\s*of\s*materials|work\s*order)\b",
    # Chemical / process engineering
    r"(?i)\b(formulation|reaction|pH|concentration|titration)\b",
    r"(?i)\b(MSDS|SDS|safety\s*data\s*sheet)\b",
    r"(?i)\b(process\s*flow|distillation|filtration|crystallization)\b",
    r"(?i)\b(soap|detergent|surfactant|saponification)\b",
    # Engineering / supply chain
    r"(?i)\b(CAD|drawing|schematic|blueprint|tolerance)\b",
    r"(?i)\b(purchase\s*order|PO|shipping\s*schedule|inventory)\b",
    r"(?i)\b(supplier\s*quality|material\s*spec|inspection\s*report)\b",
    # Regulatory / compliance (technical)
    r"(?i)\b(ISO\s*\d+|HACCP|GMP|FDA|CE\s*marking)\b",
    r"(?i)\b(audit\s*(finding|report|schedule)|corrective\s*action)\b",
]

DEFAULT_SENSITIVE_SUBJECT = [
    # HR / compensation
    r"(?i)\b(salary|compensation|bonus|raise|payroll)\b",
    r"(?i)\b(termination|layoff|fired|resignation)\b",
    r"(?i)\b(performance\s*review|PIP|disciplinary)\b",
    r"(?i)\b(confidential|private|restricted|NDA)\b",
    r"(?i)\b(medical|health|disability|leave\s*of\s*absence)\b",
    r"(?i)\b(lawsuit|legal\s*action|litigation|settlement)\b",
    r"(?i)\b(SSN|social\s*security|tax\s*ID|passport)\b",
    r"(?i)\b(bank\s*account|routing\s*number|credit\s*card)\b",
    r"(?i)\b(background\s*check|drug\s*test)\b",
    # Social / personal / greetings
    r"(?i)\b(happy\s*(birth|b[- ]?day|belated|anniversary))\b",
    r"(?i)\b(birthday|bday|b-day)\b",
    r"(?i)\b(congratulat|congrats|mabrook|mubarak)\b",
    r"(?i)\b(wedding|marriage|nikah|shaadi|engagement)\b",
    r"(?i)\b(baby\s*(born|birth|shower|arrival)|new\s*born)\b",
    r"(?i)\b(welcome\s+(aboard|to\s+the\s+team|back|new))\b",
    r"(?i)\bwelcome\s+\w+\b",  # "welcome Jane" etc.
    r"(?i)\b(farewell|goodbye|good\s*bye|leaving\s*party)\b",
    r"(?i)\b(condolence|sympathy|passed\s*away|funeral|demise)\b",
    r"(?i)\b(holiday|vacation|eid|christmas|diwali|ramadan|iftar)\b",
    r"(?i)\b(party|celebration|get[- ]?together|potluck|picnic)\b",
    r"(?i)\b(joke|funny|humor|lol|haha|forwarded?\s*joke)\b",
    r"(?i)\b(motivational|inspirational|quote\s*of\s*the\s*day)\b",
    r"(?i)\b(charity|donation|fundrais|volunteer)\b",
    r"(?i)\b(lunch|dinner|breakfast|meal\s*allowance|food\s*order)\b",
    r"(?i)\b(seating\s*plan|desk\s*allocation|office\s*move)\b",
    r"(?i)\b(out\s*of\s*office|OOO|on\s*leave|annual\s*leave)\b",
]

DEFAULT_TECHNICAL_BODY = [
    # Software / IT
    r"(?i)\b(stack\s*trace|traceback|exception|error\s*log)\b",
    r"(?i)\b(git\s*(push|pull|commit|merge|rebase))\b",
    r"(?i)\b(localhost|127\.0\.0\.1|port\s*\d{4})\b",
    r"(?i)\b(npm|pip|maven|gradle|yarn)\s+(install|update|build)\b",
    r"(?i)\b(function|class|method|variable|parameter)\b",
    r"(?i)\b(server|cluster|node|replica|shard)\b",
    r"(?i)\b(latency|throughput|SLA|uptime|availability)\b",
    r"(?i)\b(SSL|TLS|certificate|HTTPS|OAuth)\b",
    r"(?i)\b(JSON|XML|CSV|YAML|protobuf)\b",
    r"(?i)\b(query|index|cache|replication)\b",
    r"(?i)\b(CPU|RAM|memory|disk|storage)\b",
    r"(?i)\b(load\s*balancer|reverse\s*proxy|CDN)\b",
    # Manufacturing / production
    r"(?i)\b(production\s*(line|batch|schedule|run|output))\b",
    r"(?i)\b(machine\s*(setup|calibration|parameters|settings))\b",
    r"(?i)\b(quality\s*(control|check|inspection|standard))\b",
    r"(?i)\b(defect\s*(rate|analysis|report)|reject|scrap\s*rate)\b",
    r"(?i)\b(maintenance\s*(schedule|log|plan)|preventive\s*maintenance)\b",
    r"(?i)\b(raw\s*material|feedstock|raw\s*stock|ingredient)\b",
    r"(?i)\b(batch\s*(number|record|size|processing))\b",
    r"(?i)\b(yield|efficiency|OEE|cycle\s*time|takt\s*time)\b",
    # Chemical / process
    r"(?i)\b(formulation|concentration|pH\s*level|viscosity|density)\b",
    r"(?i)\b(saponification|emulsification|polymerization|catalys)\b",
    r"(?i)\b(safety\s*data\s*sheet|SDS|MSDS|hazard\s*class)\b",
    r"(?i)\b(soap|detergent|surfactant|alkali)\b",
    r"(?i)\b(distillation|evaporation|filtration|drying|mixing)\b",
    # Engineering / supply chain
    r"(?i)\b(specification|technical\s*drawing|blueprint|schematic)\b",
    r"(?i)\b(tolerance|dimension|gauge|measurement|calibration)\b",
    r"(?i)\b(purchase\s*order|supplier|vendor|procurement)\b",
    r"(?i)\b(shipping|logistics|warehouse|inventory\s*level)\b",
    # Regulatory / compliance (technical)
    r"(?i)\b(ISO\s*\d+|HACCP|GMP|OHSAS|REACH|RoHS)\b",
    r"(?i)\b(audit|non-?conformance|corrective\s*action|CAPA)\b",
    r"(?i)\b(environmental\s*(permit|compliance|report))\b",
]

DEFAULT_SENSITIVE_BODY = [
    # PII / confidential
    r"(?i)\b(social\s*security\s*number|SSN)\b",
    r"(?i)\b(date\s*of\s*birth|DOB)\b",
    r"(?i)\b(home\s*address|residential\s*address)\b",
    r"(?i)\b(please\s*keep\s*(this\s*)?confidential)\b",
    r"(?i)\b(do\s*not\s*(forward|share|distribute))\b",
    r"(?i)\bIBAN\b",
    r"(?i)\b(personal\s*(information|data|details))\b",
    # Social / personal content
    r"(?i)\b(happy\s*(birth|b[- ]?day|belated|anniversary))\b",
    r"(?i)\b(congratulat|congrats|best\s*wishes|wish\s*you)\b",
    r"(?i)\b(wedding|marriage|nikah|engagement)\b",
    r"(?i)\b(baby\s*(born|birth|shower)|new\s*born|bundle\s*of\s*joy)\b",
    r"(?i)\b(farewell|goodbye|leaving\s*party|miss\s*you)\b",
    r"(?i)\b(condolence|sympathy|passed\s*away|rest\s*in\s*peace|RIP)\b",
    r"(?i)\b(party|celebration|get[- ]?together|potluck)\b",
    r"(?i)\b(joke|funny|humor|forwarded?\s*joke|chain\s*mail)\b",
    r"(?i)\b(prayer|blessing|may\s*god|insha.?allah|god\s*bless)\b",
]

DEFAULT_TECHNICAL_FOLDERS = [
    r"(?i)(engineering|development|devops|infrastructure|platform)",
    r"(?i)(technical|IT|operations|SRE|architecture)",
    r"(?i)(sprint|project|release|deployment)",
    r"(?i)(production|manufacturing|plant|factory|quality)",
    r"(?i)(supply.?chain|logistics|procurement|maintenance)",
    r"(?i)(R&D|research|lab|process|chemical)",
]

DEFAULT_SENSITIVE_FOLDERS = [
    r"(?i)(HR|human\s*resources|people\s*ops)",
    r"(?i)(legal|compliance|finance|payroll)",
    r"(?i)(confidential|restricted|private)",
]


class EmailSensitivityClassifier:
    """
    Multi-signal email sensitivity classifier for Bronze layer.

    Scores each email across subject, content, folder, recipients,
    and attachments. Positive scores → technical, negative → sensitive.
    """

    # Signal weights (must sum to 1.0)
    WEIGHT_SUBJECT = 0.30
    WEIGHT_CONTENT = 0.30
    WEIGHT_FOLDER = 0.15
    WEIGHT_RECIPIENTS = 0.15
    WEIGHT_ATTACHMENTS = 0.10

    # How much body text to scan
    CONTENT_SAMPLE_CHARS = 3000

    def __init__(self, rules_path: str | None = None):
        """
        Initialize classifier with optional custom rules.

        Args:
            rules_path: Path to sensitivity_rules.yaml (optional).
                        Falls back to config/sensitivity_rules.yaml,
                        then to built-in defaults.
        """
        self._rules = self._load_rules(rules_path)

        # Compile regex patterns from rules (or defaults)
        self._tech_subject_re = self._compile_patterns(
            self._rules.get("technical_subject_patterns", DEFAULT_TECHNICAL_SUBJECT)
        )
        self._sens_subject_re = self._compile_patterns(
            self._rules.get("sensitive_subject_patterns", DEFAULT_SENSITIVE_SUBJECT)
        )
        self._tech_body_re = self._compile_patterns(
            self._rules.get("technical_body_patterns", DEFAULT_TECHNICAL_BODY)
        )
        self._sens_body_re = self._compile_patterns(
            self._rules.get("sensitive_body_patterns", DEFAULT_SENSITIVE_BODY)
        )
        self._tech_folder_re = self._compile_patterns(
            self._rules.get("technical_folder_patterns", DEFAULT_TECHNICAL_FOLDERS)
        )
        self._sens_folder_re = self._compile_patterns(
            self._rules.get("sensitive_folder_patterns", DEFAULT_SENSITIVE_FOLDERS)
        )

        # Manual overrides: sender email → forced classification
        self._overrides: dict[str, str] = {}
        for rule in self._rules.get("overrides", []):
            if "sender_email" in rule and "classification" in rule:
                self._overrides[rule["sender_email"].lower()] = rule["classification"]

        logger.info(
            f"EmailSensitivityClassifier initialized ({len(self._overrides)} manual overrides)"
        )

    def _load_rules(self, rules_path: str | None) -> dict:
        """Load rules from YAML file, falling back to defaults."""
        paths_to_try = []
        if rules_path:
            paths_to_try.append(Path(rules_path))

        # Default location
        config_dir = CONFIG_DIR
        paths_to_try.append(config_dir / "sensitivity_rules.yaml")

        for path in paths_to_try:
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        rules = yaml.safe_load(f) or {}
                    logger.info(f"Loaded sensitivity rules from {path}")
                    return rules
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

        logger.info("Using built-in default sensitivity rules")
        return {}

    @staticmethod
    def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
        """Compile regex patterns, skipping invalid ones."""
        compiled = []
        for p in patterns:
            try:
                compiled.append(re.compile(p))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{p}': {e}")
        return compiled

    def classify(self, email_data: dict[str, Any]) -> SensitivityResult:
        """
        Classify a single email as technical or sensitive.

        Args:
            email_data: Bronze-layer email dict with email_headers,
                        email_body_text, document_metadata, attachments.

        Returns:
            SensitivityResult with classification, confidence, signals.
        """
        headers = email_data.get("email_headers", {})
        body = email_data.get("email_body_text", "")
        meta = email_data.get("document_metadata", {})
        attachments = email_data.get("attachments", [])

        # Check manual overrides first
        sender_email = headers.get("sender_email", "").lower()
        if sender_email in self._overrides:
            forced = self._overrides[sender_email]
            return SensitivityResult(
                classification=forced,
                confidence=1.0,
                signals={"override": f"sender_email={sender_email}", "forced": forced},
            )

        signals = {}

        # Signal 1: Subject patterns
        subject = headers.get("subject", "")
        subj_score, subj_details = self._score_subject(subject)
        signals["subject"] = {"score": subj_score, **subj_details}

        # Signal 2: Body content patterns
        content_score, content_details = self._score_content(body)
        signals["content"] = {"score": content_score, **content_details}

        # Signal 3: Folder path
        folder = headers.get("folder_path", "")
        folder_score, folder_details = self._score_folder(folder)
        signals["folder"] = {"score": folder_score, **folder_details}

        # Signal 4: Recipients
        recipients = self._collect_recipients(headers)
        recip_score, recip_details = self._score_recipients(recipients)
        signals["recipients"] = {"score": recip_score, **recip_details}

        # Signal 5: Attachments
        att_score, att_details = self._score_attachments(attachments)
        signals["attachments"] = {"score": att_score, **att_details}

        # Weighted sum
        weighted_sum = (
            self.WEIGHT_SUBJECT * subj_score
            + self.WEIGHT_CONTENT * content_score
            + self.WEIGHT_FOLDER * folder_score
            + self.WEIGHT_RECIPIENTS * recip_score
            + self.WEIGHT_ATTACHMENTS * att_score
        )

        # Default to technical — enterprise emails are mostly work-related.
        # Only classify as sensitive with clear negative evidence.
        classification = "not_personal" if weighted_sum >= 0 else "personal"

        # Confidence via sigmoid
        raw = abs(weighted_sum)
        confidence = 0.5 + 0.5 * (1 - math.exp(-3 * raw))

        signals["weighted_sum"] = round(weighted_sum, 4)

        logger.debug(
            f"Email '{subject[:50]}' classified as {classification} "
            f"(confidence={confidence:.2f}, sum={weighted_sum:.3f})"
        )

        return SensitivityResult(
            classification=classification,
            confidence=round(confidence, 4),
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Signal 1: Subject patterns
    # ------------------------------------------------------------------

    def _score_subject(self, subject: str) -> tuple[float, dict[str, Any]]:
        details: dict[str, Any] = {"matched": []}
        if not subject:
            return 0.0, details

        score = 0.0
        for pat in self._tech_subject_re:
            if pat.search(subject):
                score += 0.30
                details["matched"].append(f"+tech:{pat.pattern[:40]}")
        for pat in self._sens_subject_re:
            if pat.search(subject):
                score -= 0.40
                details["matched"].append(f"-sens:{pat.pattern[:40]}")

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Signal 2: Body content
    # ------------------------------------------------------------------

    def _score_content(self, body: str) -> tuple[float, dict[str, Any]]:
        details: dict[str, Any] = {"tech_hits": 0, "sens_hits": 0}
        if not body:
            return 0.0, details

        sample = body[: self.CONTENT_SAMPLE_CHARS]
        score = 0.0

        tech_hits = 0
        for pat in self._tech_body_re:
            matches = pat.findall(sample)
            tech_hits += len(matches)
        details["tech_hits"] = tech_hits

        sens_hits = 0
        for pat in self._sens_body_re:
            matches = pat.findall(sample)
            sens_hits += len(matches)
        details["sens_hits"] = sens_hits

        # Normalize: each hit adds a diminishing amount (log scale)
        if tech_hits > 0:
            score += min(0.8, 0.15 * math.log2(tech_hits + 1))
        if sens_hits > 0:
            score -= min(0.8, 0.25 * math.log2(sens_hits + 1))

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Signal 3: Folder path
    # ------------------------------------------------------------------

    def _score_folder(self, folder: str) -> tuple[float, dict[str, Any]]:
        details: dict[str, Any] = {}
        if not folder:
            return 0.0, details

        score = 0.0
        for pat in self._tech_folder_re:
            if pat.search(folder):
                score += 0.50
                details["match"] = f"+tech:{pat.pattern[:40]}"
                break
        for pat in self._sens_folder_re:
            if pat.search(folder):
                score -= 0.60
                details["match"] = f"-sens:{pat.pattern[:40]}"
                break

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Signal 4: Recipients
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_recipients(headers: dict[str, Any]) -> list[dict[str, str]]:
        """Collect all recipients (to + cc) into a flat list."""
        recipients = []
        for field_name in ("recipients_to", "recipients_cc"):
            for r in headers.get(field_name, []):
                if isinstance(r, dict):
                    recipients.append(r)
                elif isinstance(r, str):
                    recipients.append({"email": r, "name": ""})
        return recipients

    def _score_recipients(self, recipients: list[dict[str, str]]) -> tuple[float, dict[str, Any]]:
        details: dict[str, Any] = {"count": len(recipients)}
        if not recipients:
            return 0.0, details

        score = 0.0

        # Distribution lists / team aliases → likely technical
        dl_patterns = re.compile(
            r"(?i)(engineering|dev|devops|platform|infra|sre|ops|team|all-"
            r"|production|manufacturing|quality|plant|maintenance|supply.?chain"
            r"|logistics|procurement|R&D|lab|process|technical)"
        )
        hr_patterns = re.compile(
            r"(?i)(hr|human.?resources|people|payroll|legal|compliance|finance)"
        )

        for r in recipients:
            email = r.get("email", "").lower()
            name = r.get("name", "").lower()
            combined = f"{email} {name}"

            if dl_patterns.search(combined):
                score += 0.30
                details["dl_match"] = combined[:50]
            if hr_patterns.search(combined):
                score -= 0.40
                details["hr_match"] = combined[:50]

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Signal 5: Attachments
    # ------------------------------------------------------------------

    def _score_attachments(self, attachments: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
        details: dict[str, Any] = {"count": len(attachments)}
        if not attachments:
            return 0.0, details

        score = 0.0
        tech_ext = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".go",
            ".rs",
            ".yml",
            ".yaml",
            ".json",
            ".xml",
            ".sh",
            ".tf",
            ".dockerfile",
            ".sql",
            ".log",
            ".dwg",
            ".dxf",
            ".step",
            ".stp",
            ".iges",
            ".stl",  # CAD/engineering
            ".csv",
            ".xls",
            ".xlsx",  # data/reports
        }
        sens_ext = {".pdf"}  # PDFs alone aren't sensitive, but combined with other signals

        for att in attachments:
            filename = att.get("filename", "").lower()
            ext = Path(filename).suffix.lower()
            stem = Path(filename).stem.lower()

            if ext in tech_ext:
                score += 0.40
                details["tech_file"] = filename
            # Filename keywords
            if any(
                kw in stem
                for kw in (
                    "config",
                    "deploy",
                    "pipeline",
                    "schema",
                    "migration",
                    "spec",
                    "report",
                    "batch",
                    "quality",
                    "inspection",
                    "drawing",
                    "bom",
                    "sds",
                    "msds",
                    "procedure",
                    "production",
                    "maintenance",
                    "test_result",
                )
            ):
                score += 0.30
                details["tech_keyword"] = filename
            if any(kw in stem for kw in ("payslip", "contract", "offer_letter", "nda", "tax")):
                score -= 0.50
                details["sens_keyword"] = filename

        score = max(-1.0, min(1.0, score))
        return round(score, 4), details

    # ------------------------------------------------------------------
    # Thread-level aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def should_skip_thread(
        email_classifications: list[SensitivityResult | None],
    ) -> bool:
        """
        Decide whether a thread should be skipped (personal content).

        Rule: if ANY email in the thread is classified as 'personal',
        skip the entire thread.

        Args:
            email_classifications: List of SensitivityResult (one per email
                                   in the thread). None entries are treated
                                   as 'not_personal' (safe default — process it).

        Returns:
            True if the thread should be skipped (personal),
            False if it should be processed (not_personal / work content).
        """
        for result in email_classifications:
            if result and result.classification == "personal":
                return True  # At least one personal → skip whole thread
        return False  # Default: process


class LLMSensitivityClassifier:
    """
    LLM-based email sensitivity classifier.

    Uses Azure OpenAI / OpenAI to classify emails as technical or sensitive.
    More accurate than regex but costs API calls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        use_azure: bool = False,
        azure_endpoint: str | None = None,
        azure_api_version: str = "2024-12-01-preview",
        azure_deployment: str | None = None,
        model: str = "gpt-4o",
    ):
        self.use_azure = use_azure

        try:
            if use_azure:
                from openai import AzureOpenAI

                self.model = azure_deployment or model
                self.client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    max_retries=2,
                )
            else:
                from openai import OpenAI

                self.model = model
                self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        # Load prompts
        from tacitgraph.prompt_loader import format_prompt, get_prompt

        self._get_prompt = get_prompt
        self._format_prompt = format_prompt

        logger.info(f"LLMSensitivityClassifier initialized (model={self.model})")

    def classify(self, email_data: dict[str, Any]) -> SensitivityResult:
        """
        Classify a single email using LLM.

        Args:
            email_data: Bronze-layer email dict.

        Returns:
            SensitivityResult with classification, confidence, signals.
        """
        headers = email_data.get("email_headers", {})
        body = email_data.get("email_body_text", "")
        attachments = email_data.get("attachments", [])

        subject = headers.get("subject", "(no subject)")
        sender = headers.get("sender", "") or headers.get("sender_email", "")
        recipients_to = headers.get("recipients_to", [])
        recip_str = ", ".join(
            r.get("name", r.get("email", "")) if isinstance(r, dict) else str(r)
            for r in recipients_to[:5]
        )
        folder = headers.get("folder_path", "")
        att_str = ", ".join(a.get("filename", "") for a in attachments[:5]) or "none"

        system_prompt = self._get_prompt(
            "silver",
            "sensitivity_classification",
            "system_prompt",
        )
        user_template = self._get_prompt(
            "silver",
            "sensitivity_classification",
            "user_prompt",
        )
        user_prompt = self._format_prompt(
            user_template,
            subject=subject,
            sender=sender,
            recipients=recip_str,
            folder=folder,
            body=body[:1500],
            attachments=att_str,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._get_prompt(
                    "silver", "sensitivity_classification", "temperature", 0.0
                ),
                max_tokens=self._get_prompt(
                    "silver", "sensitivity_classification", "max_tokens", 150
                ),
                response_format={"type": "json_object"},
            )

            # extract_llm_content raises LLMContentError for content_filter / length
            from tacitgraph.llm_response import extract_llm_content

            content = extract_llm_content(response, context="sensitivity classification")
            result = json.loads(content)

            classification = result.get("classification", "not_personal").lower()
            if classification not in ("personal", "not_personal"):
                classification = "not_personal"

            confidence = float(result.get("confidence", 0.7))
            reasoning = result.get("reasoning", "")

            return SensitivityResult(
                classification=classification,
                confidence=confidence,
                signals={"method": "llm", "reasoning": reasoning},
            )

        except Exception as e:
            # Handles: LLMContentError (content_filter/length), RateLimitError,
            # APIStatusError, JSONDecodeError, network timeouts, etc.
            logger.warning(f"LLM sensitivity classification failed: {e}")
            # Safe default: treat as not_personal so the email is not silently dropped
            return SensitivityResult(
                classification="not_personal",
                confidence=0.5,
                signals={"method": "llm", "error": str(e)},
            )

    @staticmethod
    def should_skip_thread(
        email_classifications: list[SensitivityResult | None],
    ) -> bool:
        """Same thread-level rule as regex classifier."""
        return EmailSensitivityClassifier.should_skip_thread(email_classifications)
