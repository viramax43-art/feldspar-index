"""Оценка качества голосовых референсов."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityReport:
    accepted: bool
    score: float
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "score": round(self.score, 2),
            "issues": self.issues,
            "recommendations": self.recommendations,
        }

    def format_message(self, index: int) -> str:
        status = "✅ принято" if self.accepted else "❌ отклонено"
        lines = [f"Референс #{index}: {status} (оценка {self.score:.0f}/100)"]
        if self.issues:
            lines.append("Проблемы: " + "; ".join(self.issues))
        if self.recommendations:
            lines.append("Рекомендации: " + "; ".join(self.recommendations))
        return "\n".join(lines)


def evaluate_reference(
    metrics: dict,
    min_duration: float = 3.0,
    max_duration: float = 60.0,
) -> QualityReport:
    issues: list[str] = []
    recommendations: list[str] = []
    score = 100.0

    duration = float(metrics.get("duration_sec", 0))
    speech_ratio = float(metrics.get("speech_ratio", 0))
    clipping = float(metrics.get("clipping_ratio", 0))
    rms = float(metrics.get("rms", 0))

    if duration < min_duration:
        issues.append(f"слишком короткий ({duration:.1f} с)")
        recommendations.append("запишите сообщение длиной 5–15 секунд")
        score -= 35
    elif duration > max_duration:
        issues.append(f"слишком длинный ({duration:.1f} с)")
        recommendations.append("разбейте на несколько коротких сообщений")
        score -= 15

    if speech_ratio < 0.45:
        issues.append(f"мало речи ({speech_ratio:.0%})")
        recommendations.append("говорите непрерывно, без длинных пауз")
        score -= 30
    elif speech_ratio < 0.6:
        score -= 10
        recommendations.append("сократите паузы между фразами")

    if clipping > 0.02:
        issues.append("обнаружен клиппинг")
        recommendations.append("отодвиньтесь от микрофона и уменьшите громкость")
        score -= 25

    if rms < 0.005:
        issues.append("слишком тихая запись")
        recommendations.append("говорите ближе к микрофону")
        score -= 30
    elif rms < 0.01:
        score -= 10
        recommendations.append("немного увеличьте громкость записи")

    if metrics.get("denoise_applied"):
        score += 2

    accepted = score >= 55 and duration >= min_duration and speech_ratio >= 0.4
    return QualityReport(
        accepted=accepted,
        score=max(0.0, min(100.0, score)),
        issues=issues,
        recommendations=recommendations,
    )


def evaluate_profile(
    references: list[dict],
    min_total_seconds: float = 30.0,
    max_total_seconds: float = 180.0,
    min_count: int = 3,
) -> QualityReport:
    accepted_refs = [r for r in references if r.get("quality", {}).get("accepted")]
    total_duration = sum(r.get("duration_sec", 0) for r in accepted_refs)
    issues: list[str] = []
    recommendations: list[str] = []
    score = 100.0

    if len(accepted_refs) < min_count:
        issues.append(f"принято только {len(accepted_refs)} из минимум {min_count}")
        recommendations.append("добавьте ещё голосовые сообщения")
        score -= 40

    if total_duration < min_total_seconds:
        issues.append(f"общая длительность {total_duration:.0f} с < {min_total_seconds:.0f} с")
        recommendations.append("запишите больше речи с разной интонацией")
        score -= 30

    if total_duration > max_total_seconds:
        issues.append(f"общая длительность {total_duration:.0f} с превышает лимит")
        score -= 10

    accepted = len(accepted_refs) >= min_count and total_duration >= min_total_seconds
    return QualityReport(
        accepted=accepted,
        score=max(0.0, min(100.0, score)),
        issues=issues,
        recommendations=recommendations,
    )
