"""Recalcula las tablas del informe desde los artefactos de la entrega, sin el corpus, sin GPU y sin LLM.

    python tools/recompute_tables.py                  # desde la raíz del repo o del zip
    python tools/recompute_tables.py --root ruta/al/zip --out tablas.md

Lee solo lo que va en el zip: ``eval/`` (el conjunto congelado) y ``artifacts/full/`` (resultados por ítem,
memoria, auditorías y ``run_report.json``). Usa solo la biblioteca estándar, a propósito: quien corrige puede
correrlo sin instalar nada y sin confiar en nuestro código de ``src/``.

Las reglas de puntaje son copias textuales de las del proyecto (``evalset.normalize``, ``evalset.contains_key``,
``grounded.verdict_of``, ``grounded.score``); ``tests/test_recompute_tables.py`` verifica que coincidan.

Qué no puede recalcular sin el corpus: la primaria de la Etapa 3 ("clave entregada desde el gold") necesita el
texto de los pasajes. La Etapa 8 la exporta por ítem a ``results/03_retrieval_metrics.jsonl``; si ese archivo
está, se usa y se compara contra ``index/best_config.json``.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ANSWER_MAX = 1200  # grounded.ANSWER_MAX: una respuesta en el tope la cortó la gramática y es un error


# --- Las reglas de puntaje, copiadas del proyecto ----------------------------------------------------

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d),0+\b", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return " ".join(text.split())


def contains_key(text: str, keys: list[str]) -> bool:
    patterns = [re.compile(r"(?<!\w)(?<!\d\.)" + re.escape(normalize(k)) + r"(?!\w)(?!\.\d)") for k in keys]
    text = normalize(text)
    return any(p.search(text) for p in patterns)


def verdict_of(answer: str) -> str:
    words = re.findall(r"\w+", unicodedata.normalize("NFC", answer).lower())
    if not words or words[:2] == ["no", "obstante"]:
        return "unclear"
    return {"sí": "yes", "no": "no"}.get(words[0], "unclear")


def correct(answer: dict | None, target: dict, call_error: str | None = None) -> bool | None:
    """``grounded.score(...)["correct"]``: None si es un error de llamada (su propia categoría)."""
    if answer is None or call_error:
        return None
    if len(answer["answer"]) >= ANSWER_MAX:
        return None
    if not target["answerable"]:
        return not answer["can_answer"]
    ok = answer["can_answer"] and contains_key(answer["answer"], target["answer_key"])
    if target.get("expected_verdict"):
        ok = ok and verdict_of(answer["answer"]) == target["expected_verdict"]
    return bool(ok)


# --- Ruido -------------------------------------------------------------------------------------------

def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def paired(a: dict, b: dict, groups: dict | None = None) -> tuple[int, int, float]:
    """Victorias y derrotas de ``b`` contra ``a`` (por ítem, o un voto por grupo) y la p del test de signos."""
    common = sorted(set(a) & set(b))
    if groups is None:
        wins = sum(b[i] and not a[i] for i in common)
        losses = sum(a[i] and not b[i] for i in common)
    else:
        delta = defaultdict(int)
        for i in common:
            delta[groups[i]] += int(b[i]) - int(a[i])
        wins, losses = sum(d > 0 for d in delta.values()), sum(d < 0 for d in delta.values())
    return wins, losses, sign_test(wins, losses)


def bootstrap_diff(a: dict, b: dict, n: int = 10_000, seed: int = 0) -> tuple[int, int]:
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    diffs = sorted(sum(b[i] - a[i] for i in (rng.choice(ids) for _ in ids)) / len(ids) for _ in range(n))
    return round(100 * diffs[int(0.025 * n)]), round(100 * diffs[int(0.975 * n)])


# --- Lectura -----------------------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Una fila por (brazo, ítem): la última escrita, como ``store.ResultsLog.rows``."""
    out = {}
    for row in rows:
        out[(str(row["arm"]), str(row["item_id"]))] = row
    return out


def read_sheet(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delimiter = max(",;\t", key=text.split("\n", 1)[0].count)
    return list(csv.DictReader(io.StringIO(text), dialect=csv.excel, delimiter=delimiter))


def frac(rows) -> str:
    rows = list(rows)
    return f"{sum(rows)}/{len(rows)}"


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def h(self, text: str) -> None:
        self.lines += ["", f"## {text}", ""]

    def p(self, text: str = "") -> None:
        self.lines.append(text)

    def table(self, header: list[str], rows: list[list]) -> None:
        if self.lines and self.lines[-1]:
            self.lines.append("")
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("|" + "---|" * len(header))
        for row in rows:
            self.lines.append("| " + " | ".join(str(x) for x in row) + " |")
        self.lines.append("")


# --- Las etapas --------------------------------------------------------------------------------------

def stage3(rep: Report, res: Path, items: dict, index_dir: Path) -> None:
    rep.h("Etapa 3 · recuperación (test, preguntas con respuesta)")
    rows = latest(read_jsonl(res / "03_retrieval.jsonl"))
    metrics_path = res / "03_retrieval_metrics.jsonl"
    metrics = latest(read_jsonl(metrics_path)) if metrics_path.is_file() else {}
    arms = list(dict.fromkeys(arm for arm, _ in rows))
    test = [i for i, t in items.items() if t["split"] == "test" and t["answerable"]]
    primary = {}
    table = []
    for arm in arms:
        doc10, sec10, tokens, n_pass = [], [], [], []
        for iid in test:
            row, gold = rows[(arm, iid)], set(items[iid]["gold_sections"])
            docs = list(dict.fromkeys(s.split("#")[0] for s in row["sections"]))
            doc10.append(bool(set(docs[:10]) & {g.split("#")[0] for g in gold}))
            sec10.append(bool(set(row["sections"][:10]) & gold))
            tokens.append(sum(n for _, n in row["delivered"]))
            n_pass.append(len(row["delivered"]))
        key = "—"
        if metrics:
            primary[arm] = {iid: bool(metrics[(arm, iid)]["metrics"]["gold_key_delivered"]) for iid in test}
            key = frac(primary[arm].values())
        ms = sorted(rows[(arm, iid)]["latency_ms"]["total"] for iid in test)
        table.append([arm, frac(doc10), frac(sec10), key, f"{sum(n_pass) / len(n_pass):.1f}",
                      round(sum(tokens) / len(tokens)), round(ms[len(ms) // 2])])
    rep.table(["fila", "doc@10", "sección@10 (estricto, por id)", "clave desde gold", "pasajes", "tokens",
               "ms (mediana)"], table)
    rep.p("`sección@10` cuenta acá solo el id gold; la tabla del informe acepta además las versiones del mismo "
          "bloque con texto idéntico, y eso necesita el corpus.")
    if not metrics:
        rep.p("`clave desde gold`: falta `results/03_retrieval_metrics.jsonl` (lo escribe la Etapa 8, que lee el corpus).")
        return
    names = [a for a in arms if a != "dense·A/A"]
    noise = []
    for prev, cur in zip(names, names[1:]):
        w, l, p = paired(primary[prev], primary[cur])
        noise.append([f"{cur} vs {prev}", f"{sum(primary[prev].values())} → {sum(primary[cur].values())}", w, l,
                      round(p, 3)])
    rep.table(["comparación", "aciertos", "victorias", "derrotas", "p (signos)"], noise)
    best_path = index_dir / "best_config.json"
    if best_path.is_file():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        same = all(best["primary_on_test"][a] == sum(primary[a].values()) for a in best["primary_on_test"])
        rep.p(f"Contra `best_config.json`: {'coinciden' if same else 'NO coinciden'} los aciertos de las {len(best['primary_on_test'])} filas.")


def stage4(rep: Report, res: Path, items: dict) -> None:
    rep.h("Etapa 4 · respuestas fundamentadas (test; con los brazos de la Etapa 8, si están)")
    rows = latest(read_jsonl(res / "04_grounded_answers.jsonl"))
    if (res / "08_grounded_answers.jsonl").is_file():
        rows |= latest(read_jsonl(res / "08_grounded_answers.jsonl"))
    arms = list(dict.fromkeys(arm for arm, _ in rows))
    scored, mismatches = {}, 0
    for (arm, iid), row in rows.items():
        mine = correct(row["answer"], items[iid], row.get("error"))
        mismatches += mine != row["score"].get("correct")
        if items[iid]["split"] == "test" and mine is not None:
            scored.setdefault(arm, {})[iid] = mine
    table = []
    for arm in arms:
        s = scored.get(arm, {})
        ans = [s[i] for i in s if items[i]["answerable"]]
        ooc = [s[i] for i in s if not items[i]["answerable"]]
        n_test = sum(1 for (a, i) in rows if a == arm and items[i]["split"] == "test")
        table.append([arm, frac(s.values()), frac(ans), frac(ooc), n_test - len(s)])
    rep.table(["brazo", "acierto", "con respuesta", "abstención correcta", "errores"], table)
    pairs = [(a, b) for a, b in [("closed_book", "retrieval"), ("retrieval", "oracle"), ("retrieval", "retrieval_date"),
                                 ("oracle", "oracle_padded"), ("retrieval", "oracle_padded")]
             if a in scored and b in scored]
    noise = []
    for a, b in pairs:
        w, l, p = paired(scored[a], scored[b])
        lo, hi = bootstrap_diff(scored[a], scored[b])
        noise.append([f"{b} vs {a}", w, l, round(p, 3), f"{lo:+d} a {hi:+d}"])
    rep.table(["comparación", "gana B", "gana A", "p (signos)", "IC95 diferencia (pp)"], noise)
    with_passages = [r for (a, i), r in rows.items() if a != "closed_book" and items[i]["split"] == "test"
                     and r["score"].get("cited_not_delivered") is not None]
    gaps = Counter(r["call"]["prompt_tokens"] - r["call"]["expected_prompt_tokens"] for r in rows.values()
                   if r.get("call") and r["call"].get("prompt_tokens") and r["call"].get("expected_prompt_tokens"))
    rep.p(f"Re-puntuadas: {len(rows)} filas, {mismatches} distintas del puntaje guardado. "
          f"Citó sin recibir: {sum(bool(r['score']['cited_not_delivered']) for r in with_passages)} de {len(with_passages)}. "
          f"Servidor − nuestra cuenta: {dict(gaps)} tokens.")


def stage5(rep: Report, res: Path, dialogues: list[dict]) -> None:
    rep.h("Etapa 5A · con qué buscar (turnos con respuesta de test)")
    split = {d["dialogue_id"]: d["split"] for d in dialogues}
    q = [r for r in read_jsonl(res / "05_queries.jsonl")]
    agg = defaultdict(Counter)
    for r in latest(q).values():
        if r["split"] != "test" or not r.get("metrics"):
            continue
        m = r["metrics"]
        agg[r["arm"]]["n"] += 1
        agg[r["arm"]]["doc@5"] += (m.get("doc_rank") or 99) <= 5
        agg[r["arm"]]["sección@5"] += (m.get("section_rank") or 99) <= 5
        agg[r["arm"]]["clave desde gold"] += bool(m.get("gold_key_delivered"))
    rep.table(["consulta", "doc@5", "sección@5", "clave desde gold"],
              [[a, f"{c['doc@5']}/{c['n']}", f"{c['sección@5']}/{c['n']}", f"{c['clave desde gold']}/{c['n']}"]
               for a, c in agg.items()])

    rep.h("Etapa 5B · qué conservar (sobre los turnos que puntuaron en todos los brazos)")
    turns = {}
    for d in dialogues:
        for n, t in enumerate(d["turns"], start=1):
            turns[f"{d['dialogue_id']}.t{n}"] = t
    for name in ("05_history.jsonl", "08_history.jsonl"):
        path = res / name
        if not path.is_file():
            continue
        h = latest(read_jsonl(path))
        by_arm, groups = defaultdict(dict), {}
        for (arm, tid), r in h.items():
            groups[tid] = r["dialogue_id"]
            if split[r["dialogue_id"]] != "test":
                continue
            c = correct(r["answer"], turns[tid], r.get("error"))
            if c is not None:
                by_arm[arm][tid] = c
        common = set.intersection(*(set(v) for v in by_arm.values())) if by_arm else set()
        rep.p(f"`{name}`: {len(common)} turnos comunes.")
        rep.table(["brazo", "acierto (comunes)", "acierto (propios)"],
                  [[a, frac(v[i] for i in common), frac(v.values())] for a, v in by_arm.items()])
        if name == "05_history.jsonl":
            noise = []
            for a, b in [("window_h100", "window_h150"), ("window_h150", "window_h250"),
                         ("window_h100", "window_h250"), ("window_h150", "summary_h150")]:
                ca, cb = ({i: by_arm[x][i] for i in common} for x in (a, b))
                w, l, p = paired(ca, cb, groups)
                noise.append([f"{b} vs {a}", w, l, round(p, 3)])
            rep.table(["comparación", "guiones que gana B", "que gana A", "p (signos, por guion)"], noise)


def stage6(rep: Report, res: Path, scenarios: list[dict]) -> None:
    rep.h("Etapa 6 · memoria entre sesiones (sondas de test, conteos)")
    probes = {p["probe_id"]: p for s in scenarios for p in s["probes"]}
    split = {s["scenario_id"]: s["split"] for s in scenarios}
    rows = latest(read_jsonl(res / "06_memory.jsonl"))
    by = defaultdict(Counter)
    per = defaultdict(dict)
    groups = {}
    for (arm, pid), r in rows.items():
        if split[r["scenario_id"]] != "test":
            continue
        c = correct(r["answer"], probes[pid], r.get("error"))
        by[arm][r["ability"]] += bool(c)
        by[arm]["total"] += bool(c)
        per[arm][pid] = bool(c)
        groups[pid] = r["scenario_id"]
    abilities = ["extraction", "multi_session", "temporal", "knowledge_update", "abstention", "preference", "total"]
    arms = list(by)
    rep.table(["habilidad"] + arms, [[ab] + [by[a][ab] for a in arms] for ab in abilities])
    noise = []
    for a, b in [("no_memory", "replay"), ("no_memory", "store"), ("replay", "store"), ("replay_600", "store")]:
        if a in per and b in per:
            w, l, p = paired(per[a], per[b])
            gw, gl, gp = paired(per[a], per[b], groups)
            noise.append([f"{b} vs {a}", f"{w}-{l}", round(p, 3), f"{gw}-{gl}", round(gp, 3)])
    rep.table(["comparación", "por sonda", "p", "por escenario", "p"], noise)


def stage8_position(rep: Report, res: Path, items: dict) -> None:
    rep.h("Etapa 8 · posición del pasaje gold (test, preguntas con respuesta)")
    rows = latest(read_jsonl(res / "08_position.jsonl"))
    by = defaultdict(dict)
    for (arm, iid), r in rows.items():
        c = correct(r["answer"], items[iid], r.get("error"))
        if c is not None:
            by[arm][iid] = c
    rep.table(["posición", "acierto"], [[a.removeprefix("pos_"), frac(v.values())] for a, v in by.items()])
    noise = []
    for a, b in [("pos_first", "pos_middle"), ("pos_last", "pos_middle"), ("pos_first", "pos_last")]:
        if a in by and b in by:
            w, l, p = paired(by[a], by[b])
            noise.append([f"{b.removeprefix('pos_')} vs {a.removeprefix('pos_')}", w, l, round(p, 3)])
    rep.table(["comparación", "gana B", "gana A", "p (signos)"], noise)


def stage7(rep: Report, art: Path) -> None:
    rep.h("Etapa 7a · el juez contra la auditoría")
    res, judge_dir = art / "results", art / "judge"
    verdicts = {}
    for path in sorted(res.glob("07_judge_*.jsonl")):
        name = path.stem.removeprefix("07_judge_")
        verdicts[name] = {r["uid"]: r for r in read_jsonl(path)}
    if not verdicts or not (judge_dir / "audit_key.jsonl").is_file():
        rep.p("No hay veredictos del juez o falta la planilla.")
        return
    first = next(iter(verdicts.values()))
    key = {r["audit_id"]: r for r in read_jsonl(judge_dir / "audit_key.jsonl")}

    def instruments(uid: str) -> dict[str, bool]:
        out = {"contención": bool(first[uid]["contained"])}
        out |= {f"juez {name}": v[uid]["judge_verdict"] == "correct" for name, v in verdicts.items()}
        return out

    sheets = {"auditor-modelo (judge/audit.csv)": judge_dir / "audit.csv",
              "persona (judge_human/audit.csv)": art / "judge_human" / "audit.csv"}
    reference = {}
    for label, path in sheets.items():
        if not path.is_file():
            continue
        filled = {r["audit_id"]: r["human_verdict"].strip().lower() for r in read_sheet(path)
                  if r.get("human_verdict", "").strip()}
        if not filled:
            rep.p(f"{label}: sin completar ({path.name}).")
            continue
        reference[label] = filled
        table = []
        for name in instruments(key[next(iter(filled))]["uid"]):
            agree = fa = fr = 0
            for aid, verdict in filled.items():
                says = instruments(key[aid]["uid"])[name]
                truth = verdict == "correct"
                agree += says == truth
                fa += says and not truth
                fr += truth and not says
            table.append([name, f"{agree}/{len(filled)}", fa, fr])
        rep.p(f"Contra {label}:")
        rep.table(["instrumento", "acuerdo (muestra)", "acepta lo que la referencia rechaza",
                   "rechaza lo que la referencia acepta"], table)
    if len(reference) == 2:
        a, b = reference.values()
        both = set(a) & set(b)
        rep.p(f"Persona contra auditor-modelo: coinciden en {sum(a[i] == b[i] for i in both)} de {len(both)}.")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--profile", default="full")
    ap.add_argument("--out", default=None, help="además, escribir las tablas en este archivo markdown")
    args = ap.parse_args()
    root = Path(args.root)
    art = root / "artifacts" / args.profile
    eval_dir = root / "eval"
    items = {r["item_id"]: r for r in read_jsonl(eval_dir / "single_turn.jsonl")}
    dialogues = read_jsonl(eval_dir / "dialogues.jsonl")
    scenarios = read_jsonl(eval_dir / "scenarios.jsonl")
    frozen = json.loads((eval_dir / "FROZEN.json").read_text(encoding="utf-8"))
    import hashlib
    changed = [name for name, sha in frozen["files"].items()
               if hashlib.sha256((eval_dir / name).read_bytes()).hexdigest() != sha]

    rep = Report()
    rep.p("# Tablas recalculadas desde los artefactos")
    rep.p(f"Eval set `{frozen['key']}`, congelado el {frozen['frozen_at']}: "
          + ("los tres archivos coinciden con sus hashes." if not changed else f"CAMBIARON {changed}."))
    res = art / "results"
    stages = [("03_retrieval.jsonl", lambda: stage3(rep, res, items, art / "index")),
              ("04_grounded_answers.jsonl", lambda: stage4(rep, res, items)),
              ("05_queries.jsonl", lambda: stage5(rep, res, dialogues)),
              ("06_memory.jsonl", lambda: stage6(rep, res, scenarios)),
              ("08_position.jsonl", lambda: stage8_position(rep, res, items))]
    for name, run in stages:
        if (res / name).is_file():
            run()
        else:
            rep.p(f"\n(sin `results/{name}`: esa etapa no se recalcula)")
    stage7(rep, art)
    text = "\n".join(rep.lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
