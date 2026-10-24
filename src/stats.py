"""Cotas de ruido: comparaciones pareadas, test de signos y bootstrap (Etapas 4-7).

Con n = 30 una fila tiene unos ±12 puntos de ruido (assignment, Etapa 3), así que una diferencia entre
dos brazos se lee primero como conteo: cuántas preguntas cambiaron de resultado y para qué lado. Solo las
discordantes llevan información, y el test exacto de McNemar es el test de signos sobre ellas: es una
sola función.

El bootstrap remuestrea **grupos** cuando se le pasan: en la Etapa 5 los turnos de un mismo guion no son
independientes (assignment, Etapa 5: "remuestreá guiones, no turnos").
"""
import math
import random
from collections.abc import Mapping


def sign_test(wins: int, losses: int) -> float:
    """p bilateral exacto del test de signos (= McNemar exacto). Un 5-0 da 0,0625; un 6-0, 0,03125."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: Mapping[str, bool], b: Mapping[str, bool]) -> dict:
    """``b`` contra ``a`` sobre los ids puntuados en los dos: victorias de ``b``, derrotas y empates.

    Un id que falta en un brazo (un error de llamada) no entra: un error no es ni acierto ni fallo.
    """
    common = sorted(set(a) & set(b))
    wins = sum(bool(b[i]) and not a[i] for i in common)
    losses = sum(bool(a[i]) and not b[i] for i in common)
    return {"n": len(common), "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
            "p_sign": sign_test(wins, losses)}


def paired_groups(a: Mapping[str, bool], b: Mapping[str, bool], groups: Mapping[str, str]) -> dict:
    """El test de signos a nivel de grupo (guion, escenario): un grupo gana si ``b`` acierta más ítems que ``a``.

    Con turnos de un mismo guion, el test por turno (``paired``) los trata como independientes y da una p
    demasiado chica. Este cuenta un voto por guion: con 8 guiones, el mejor resultado posible es 8-0 (p = 0,008).
    """
    common = sorted(set(a) & set(b))
    by_group: dict[str, int] = {}
    for i in common:
        by_group[groups[i]] = by_group.get(groups[i], 0) + int(bool(b[i])) - int(bool(a[i]))
    wins = sum(d > 0 for d in by_group.values())
    losses = sum(d < 0 for d in by_group.values())
    return {"groups": len(by_group), "wins": wins, "losses": losses, "ties": len(by_group) - wins - losses,
            "p_sign": sign_test(wins, losses)}


def _by_group(ids: list[str], groups: Mapping[str, str] | None) -> list[list[str]]:
    if groups is None:
        return [[i] for i in ids]
    clusters: dict[str, list[str]] = {}
    for i in ids:
        clusters.setdefault(groups[i], []).append(i)
    return [clusters[g] for g in sorted(clusters)]


def _percentiles(samples: list[float], level: float) -> tuple[float, float]:
    samples = sorted(samples)
    lo = samples[int((1 - level) / 2 * (len(samples) - 1))]
    hi = samples[int((1 + level) / 2 * (len(samples) - 1))]
    return lo, hi


def bootstrap_ci(values: Mapping[str, float], groups: Mapping[str, str] | None = None,
                 n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(media, cota inferior, cota superior) con un bootstrap percentil.

    Con ``groups`` (id -> grupo) se remuestrean grupos enteros y la media es sobre los ids de los grupos
    sorteados, con repetición: un guion sorteado dos veces cuenta sus turnos dos veces.
    """
    ids = sorted(values)
    if not ids:
        return math.nan, math.nan, math.nan
    clusters = _by_group(ids, groups)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        drawn = [i for _ in clusters for i in rng.choice(clusters)]
        means.append(sum(float(values[i]) for i in drawn) / len(drawn))
    mean = sum(float(values[i]) for i in ids) / len(ids)
    return (mean, *_percentiles(means, level))


def bootstrap_diff_ci(a: Mapping[str, float], b: Mapping[str, float], groups: Mapping[str, str] | None = None,
                      n_boot: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(media de b − media de a, cotas) pareado: cada remuestra toma los mismos ids en los dos brazos."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return math.nan, math.nan, math.nan
    clusters = _by_group(ids, groups)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        drawn = [i for _ in clusters for i in rng.choice(clusters)]
        diffs.append(sum(float(b[i]) - float(a[i]) for i in drawn) / len(drawn))
    diff = sum(float(b[i]) - float(a[i]) for i in ids) / len(ids)
    return (diff, *_percentiles(diffs, level))


def count(values: Mapping[str, bool]) -> str:
    """'7/30': los conteos se reportan como conteos (assignment, Etapa 6), no como porcentajes sueltos."""
    return f"{sum(bool(v) for v in values.values())}/{len(values)}"
