"""Deterministic accessibility audit helpers for ACCS."""

from __future__ import annotations

from typing import Any

from .models import AccessibilityStandard, AccessibilityTarget


class AccessibilityAuditEngine:
	"""Evaluate accessibility targets against local, deterministic checks."""

	def audit_target(
		self,
		standard: AccessibilityStandard,
		target: AccessibilityTarget,
	) -> tuple[dict[str, Any], ...]:
		findings: list[dict[str, Any]] = []
		if target.contrast_ratio < 4.5:
			findings.append({
				"rule": "published_ui_requires_contrast",
				"severity": "high" if target.published_ui else "medium",
				"description": f"{target.surface} contrast ratio {target.contrast_ratio:.2f} is below {standard.name} {standard.version} {standard.level}.",
				"evidence": {"contrast_ratio": target.contrast_ratio, "required_ratio": 4.5},
			})
		if not target.semantic_labels_present:
			findings.append({
				"rule": "semantic_labels_required",
				"severity": "medium",
				"description": f"{target.surface} is missing semantic labels for assistive technologies.",
				"evidence": {"semantic_labels_present": False},
			})
		if not target.keyboard_navigation_present:
			findings.append({
				"rule": "keyboard_navigation_required",
				"severity": "critical",
				"description": f"{target.surface} cannot be operated with keyboard navigation.",
				"evidence": {"keyboard_navigation_present": False},
			})
		if target.media_content_present and not target.captions_available:
			findings.append({
				"rule": "media_requires_captions",
				"severity": "high",
				"description": f"{target.surface} includes media without captions or transcript.",
				"evidence": {"media_content_present": True, "captions_available": False},
			})
		return tuple(findings)

	def summarize_findings(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
		by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
		for finding in findings:
			severity = str(finding.get("severity") or "low")
			by_severity[severity] = by_severity.get(severity, 0) + 1
		return {
			"finding_count": len(findings),
			"by_severity": by_severity,
			"critical_or_high_count": by_severity.get("critical", 0) + by_severity.get("high", 0),
		}
