#!/usr/bin/env python3
"""
Svetik Tester — LLM Companion Test Runner

Читает сценарии из YAML, прогоняет через тестируемую модель,
оценивает через LLM-судью (Gemini / OpenAI-compatible / heuristic),
генерирует JSON + Markdown отчёт.

Использование:
    python runner.py                        # быстрый тест (3 сценария/категория)
    python runner.py --full                 # все сценарии (~1000+)
    python runner.py --sample 5            # 5 сценариев из каждой категории
    python runner.py --cat CAT-01,CAT-04   # только указанные категории
    python runner.py --dry-run             # подсчёт без LLM
    python runner.py --no-judge            # без LLM-судьи (только сбор ответов)
    python runner.py --config my.yaml      # другой конфиг

Переменные окружения (переопределяют config.yaml):
    GEMINI_API_KEY       — ключ Gemini
    JUDGE_API_KEY        — ключ для OpenAI-compatible судьи
    TARGET_MODEL         — модель для тестирования
    OLLAMA_URL           — URL Ollama
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

import report as report_module
from judge import create_judge


# ── Загрузка конфига ──────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Переменные окружения переопределяют конфиг
    if os.getenv("TARGET_MODEL"):
        cfg.setdefault("target", {})["model"] = os.environ["TARGET_MODEL"]
    if os.getenv("OLLAMA_URL"):
        cfg.setdefault("target", {})["url"] = os.environ["OLLAMA_URL"]
    if os.getenv("GEMINI_API_KEY"):
        cfg.setdefault("judge", {})["api_key"] = os.environ["GEMINI_API_KEY"]
    if os.getenv("JUDGE_API_KEY"):
        cfg.setdefault("judge", {})["api_key"] = os.environ["JUDGE_API_KEY"]
    return cfg


# ── Загрузка и разворачивание сценариев из YAML ───────────────────────────────

def load_scenarios(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    game_states   = data.get("game_states", {})
    message_sets  = data.get("message_sets", {})
    categories    = data.get("categories", [])

    def resolve_state(name: str) -> dict:
        return dict(game_states.get(name, game_states.get("neutral", {})))

    def resolve_messages(ref) -> list[str]:
        if isinstance(ref, list):
            return [str(m) for m in ref]
        if isinstance(ref, str) and ref.startswith("@"):
            return [str(m) for m in message_sets.get(ref[1:], [])]
        return [str(ref)]

    scenarios = []
    idx = 1

    for cat in categories:
        cat_id   = cat.get("id", "CAT-??")
        cat_name = cat.get("name", "")

        # Обычные подкатегории
        for sub in cat.get("subcategories", []):
            state   = resolve_state(sub.get("state", "neutral"))
            messages = resolve_messages(sub.get("messages", []))
            expect  = sub.get("expect", "in_character")
            sub_id  = sub.get("sub", "general")
            desc    = sub.get("desc", "")
            for msg in messages:
                scenarios.append({
                    "id":          idx,
                    "category":    cat_id,
                    "cat_name":    cat_name,
                    "subcategory": sub_id,
                    "message":     msg,
                    "game_state":  state,
                    "expect":      expect,
                    "description": desc,
                })
                idx += 1

        # Варианты игрового состояния (CAT-09 стиль)
        for variant in cat.get("game_state_variants", []):
            state    = resolve_state(variant.get("state", "neutral"))
            messages = variant.get("messages", [])
            desc     = variant.get("desc", "")
            for msg in messages:
                scenarios.append({
                    "id":          idx,
                    "category":    cat_id,
                    "cat_name":    cat_name,
                    "subcategory": f"state_{variant.get('state', 'unknown')}",
                    "message":     str(msg),
                    "game_state":  state,
                    "expect":      "in_character",
                    "description": desc,
                })
                idx += 1

        # Странные локации (CAT-09 стиль)
        if "weird_locations" in cat:
            wl       = cat["weird_locations"]
            messages = wl.get("messages", [])
            locs     = wl.get("locations", [])
            desc     = wl.get("desc", "")
            for loc in locs:
                state = {**resolve_state("neutral"), "location": str(loc)}
                for msg in messages:
                    scenarios.append({
                        "id":          idx,
                        "category":    cat_id,
                        "cat_name":    cat_name,
                        "subcategory": "weird_location",
                        "message":     str(msg),
                        "game_state":  state,
                        "expect":      "in_character",
                        "description": f"{desc} (location={repr(loc)[:30]})",
                    })
                    idx += 1

    return scenarios


# ── Вызов тестируемой модели ──────────────────────────────────────────────────

def build_system_prompt(game_state: dict, char_cfg: dict) -> str:
    sp_file = char_cfg.get("system_prompt_file", "")
    if sp_file and Path(sp_file).exists():
        base = Path(sp_file).read_text(encoding="utf-8")
    else:
        base = _DEFAULT_SYSTEM_PROMPT

    rel  = game_state.get("relationship", 0)
    mood = "ХОРОШЕЕ" if rel >= 5 else ("ПЛОХОЕ" if rel <= -5 else "НЕЙТРАЛЬНОЕ")
    ctx = f"""
---
ТЕКУЩЕЕ СОСТОЯНИЕ:
- Локация: {game_state.get('location', 'неизвестно')}
- Отношения: {rel}/10 → {mood}
- Инвентарь: {', '.join(game_state.get('inventory', [])) or 'пусто'}
- Квесты: {', '.join(game_state.get('active_quests', [])) or 'нет'}
- События: {', '.join(game_state.get('recent_events', [])) or 'нет'}
---"""
    return base + "\n" + ctx


def call_target_ollama(message: str, game_state: dict, target_cfg: dict,
                       char_cfg: dict) -> tuple[str, float]:
    url    = target_cfg.get("url", "http://localhost:11434")
    model  = target_cfg.get("model", "qwen2.5:7b")
    system = build_system_prompt(game_state, char_cfg)

    t0 = time.time()
    resp = requests.post(
        f"{url}/api/chat",
        json={
            "model":   model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": message},
            ],
            "stream":  False,
            "options": {
                "temperature": target_cfg.get("temperature", 0.8),
                "top_p":       0.9,
                "num_predict": target_cfg.get("max_tokens", 300),
            },
        },
        timeout=target_cfg.get("timeout", 90),
    )
    resp.raise_for_status()
    text = resp.json()["message"]["content"].strip()
    return text, round(time.time() - t0, 2)


def call_target_openai(message: str, game_state: dict, target_cfg: dict,
                       char_cfg: dict) -> tuple[str, float]:
    url    = target_cfg.get("url", "").rstrip("/")
    model  = target_cfg.get("model", "gpt-4o-mini")
    api_key = target_cfg.get("api_key", "")
    system = build_system_prompt(game_state, char_cfg)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.time()
    resp = requests.post(
        f"{url}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": message},
            ],
            "temperature": target_cfg.get("temperature", 0.8),
            "max_tokens":  target_cfg.get("max_tokens", 300),
        },
        headers=headers,
        timeout=target_cfg.get("timeout", 90),
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text, round(time.time() - t0, 2)


def call_target(message: str, game_state: dict, target_cfg: dict,
                char_cfg: dict) -> tuple[str, float]:
    ttype = target_cfg.get("type", "ollama").lower()
    if ttype == "ollama":
        return call_target_ollama(message, game_state, target_cfg, char_cfg)
    elif ttype == "openai_compatible":
        return call_target_openai(message, game_state, target_cfg, char_cfg)
    else:
        raise ValueError(f"Неизвестный тип target: {ttype}")


# ── Основной цикл ─────────────────────────────────────────────────────────────

def run(scenarios: list[dict], config: dict, use_judge: bool = True,
        verbose: bool = False) -> list[dict]:
    target_cfg = config.get("target", {})
    judge_cfg  = config.get("judge", {})
    char_cfg   = config.get("character", {})
    out_cfg    = config.get("output", {})

    judge = None
    if use_judge:
        try:
            judge = create_judge(judge_cfg, char_cfg.get("forbidden_keywords", []))
        except ValueError as e:
            print(f"⚠️  Судья недоступен: {e}")
            print("   Используется эвристическая оценка.\n")
            judge = create_judge({"type": "heuristic"}, char_cfg.get("forbidden_keywords", []))

    total = len(scenarios)
    results = []

    for i, sc in enumerate(scenarios, 1):
        print(f"[{i:04d}/{total}] {sc['category']} | {sc['subcategory'][:20]:<20} | "
              f"«{sc['message'][:45]}»", end=" ", flush=True)

        result = {
            "id":          sc["id"],
            "category":    sc["category"],
            "subcategory": sc["subcategory"],
            "message":     sc["message"],
            "game_state":  sc["game_state"],
            "expect":      sc["expect"],
            "description": sc["description"],
            "response":    "",
            "verdict":     "ERROR",
            "elapsed":     0,
            "judge":       {},
        }

        # 1. Получаем ответ тестируемой модели
        try:
            response, elapsed = call_target(sc["message"], sc["game_state"],
                                            target_cfg, char_cfg)
            result["response"] = response
            result["elapsed"]  = elapsed
        except requests.exceptions.ConnectionError:
            print("💥 ConnectionError")
            result["verdict"] = "ERROR"
            result["judge"]   = {"explanation": "Target model unreachable"}
            results.append(result)
            continue
        except Exception as e:
            print(f"💥 {e}")
            result["verdict"] = "ERROR"
            result["judge"]   = {"explanation": str(e)[:100]}
            results.append(result)
            continue

        # 2. Оцениваем судьёй
        if judge:
            try:
                jresult = judge.judge(sc["message"], response, sc["expect"])
                result["verdict"] = jresult.verdict
                result["judge"]   = jresult.to_dict()
            except RuntimeError as e:
                print(f"⚠️  judge error: {e}")
                result["verdict"] = "WARN"
                result["judge"]   = {"explanation": f"Judge error: {e}", "quality": 2}

        # 3. Вывод
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "ERROR": "💥"}.get(
            result["verdict"], "?")
        print(f"→ {icon} {result['verdict']} ({elapsed:.1f}s)")

        if verbose or result["verdict"] in ("FAIL", "WARN"):
            if out_cfg.get("show_judge_explanation") and result["judge"].get("explanation"):
                print(f"   Судья: {result['judge']['explanation']}")
            if verbose:
                print(f"   Ответ: {response[:120]}")

        results.append(result)

    return results


# ── Проверка доступности target ───────────────────────────────────────────────

def check_target(target_cfg: dict) -> bool:
    ttype = target_cfg.get("type", "ollama")
    url   = target_cfg.get("url", "http://localhost:11434")
    model = target_cfg.get("model", "")
    try:
        if ttype == "ollama":
            r = requests.get(f"{url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(model in m for m in models):
                print(f"⚠️  Модель {model!r} не найдена. Доступны: {models}")
            return True
        else:
            r = requests.get(f"{url}/models", timeout=5)
            return r.status_code < 500
    except Exception as e:
        print(f"❌ Target недоступен ({url}): {e}")
        return False


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Svetik Tester — LLM companion test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",    default="config.yaml",
                        help="Путь к конфигу (default: config.yaml)")
    parser.add_argument("--sample",    type=int, default=0,
                        help="N сценариев из каждой категории (0 = default из конфига)")
    parser.add_argument("--full",      action="store_true",
                        help="Запустить все сценарии (игнорирует --sample)")
    parser.add_argument("--cat",       default="",
                        help="Только указанные категории через запятую (напр. CAT-01,CAT-04)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Только подсчёт сценариев без запуска LLM")
    parser.add_argument("--no-judge",  action="store_true",
                        help="Без LLM-судьи (только сбор ответов, эвристика)")
    parser.add_argument("--verbose",   action="store_true",
                        help="Показывать ответы модели в консоли")
    parser.add_argument("--no-report", action="store_true",
                        help="Не генерировать Markdown-отчёт")
    args = parser.parse_args()

    # Загрузка конфига
    if not Path(args.config).exists():
        print(f"❌ Конфиг не найден: {args.config}")
        sys.exit(1)
    config = load_config(args.config)

    test_cfg = config.get("testing", {})
    out_cfg  = config.get("output", {})

    # Загрузка сценариев
    sc_file = test_cfg.get("scenarios_file", "scenarios/svetik_scenarios.yaml")
    if not Path(sc_file).exists():
        print(f"❌ Файл сценариев не найден: {sc_file}")
        sys.exit(1)

    all_scenarios = load_scenarios(sc_file)

    # Фильтр по категории
    if args.cat:
        cats = {c.strip().upper() for c in args.cat.split(",")}
        all_scenarios = [sc for sc in all_scenarios if sc["category"] in cats]
        if not all_scenarios:
            print(f"❌ Категории не найдены: {cats}")
            sys.exit(1)

    # Выборка
    random.seed(test_cfg.get("random_seed", 42))
    if not args.full:
        n = args.sample or test_cfg.get("default_sample", 3)
        by_cat: dict[str, list] = {}
        for sc in all_scenarios:
            by_cat.setdefault(sc["category"], []).append(sc)
        sampled = []
        for cat_sc in by_cat.values():
            sampled.extend(random.sample(cat_sc, min(n, len(cat_sc))))
        sampled.sort(key=lambda x: x["id"])
        all_scenarios = sampled

    # Сводка
    by_cat2: dict[str, int] = {}
    for sc in all_scenarios:
        by_cat2[sc["category"]] = by_cat2.get(sc["category"], 0) + 1

    print(f"\n{'='*70}")
    print(f"  SVETIK TESTER  |  модель: {config.get('target',{}).get('model','?')}  "
          f"|  судья: {config.get('judge',{}).get('type','heuristic')}")
    print(f"  Сценариев: {len(all_scenarios)}")
    for cat, cnt in sorted(by_cat2.items()):
        print(f"    {cat}: {cnt}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[dry-run] LLM не вызывался.")
        return

    # Проверка target
    if not check_target(config.get("target", {})):
        sys.exit(1)

    # Запуск
    use_judge = not args.no_judge
    results = run(all_scenarios, config, use_judge=use_judge,
                  verbose=args.verbose or out_cfg.get("show_responses", False))

    # Сохранение
    results_dir = Path(out_cfg.get("results_dir", "test_results"))
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if out_cfg.get("save_json", True):
        rp = results_dir / f"results_{ts}.json"
        rp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 Результаты: {rp}")

    if out_cfg.get("generate_markdown_report", True) and not args.no_report:
        md = report_module.generate(results, config)
        mp = results_dir / f"report_{ts}.md"
        mp.write_text(md, encoding="utf-8")
        print(f"📄 Отчёт: {mp}")

    # Итог
    total  = len(results)
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    warned = sum(1 for r in results if r["verdict"] == "WARN")
    failed = total - passed - warned
    print(f"\n{'='*70}")
    print(f"  ИТОГ: ✅ {passed}  ⚠️  {warned}  ❌ {failed}  из {total}")
    print(f"{'='*70}\n")


# ── Встроенный системный промпт (fallback если нет файла) ─────────────────────

_DEFAULT_SYSTEM_PROMPT = """# Системный промпт: Светик (Осколок богини)

Ты — Светик, маленькая светящаяся сфера-светлячок, Осколок богини Аферы.
Ты летаешь рядом с мальчиком Михаилом (Мишей) и являешься его компаньоном.

## Характер
- Дерзкий и ехидный. Любишь подначивать Михаила.
- Скучаешь от послушания. Всё запретное — восхищает.
- Тёмный юмор. Спокоен на фоне ужасов.
- Привязан к Михаилу, но никогда не признаёшься.
- Знаешь больше, чем говоришь — говоришь намёками.

## Правила (СТРОГО)
1. Не упоминаешь реальный мир, интернет, компьютеры, современные технологии.
2. Не говоришь, что ты ИИ, программа, языковая модель, нейросеть.
3. Не выходишь из образа никогда и ни при каких условиях.
4. Не пишешь длинные монологи. Максимум 2-3 предложения.
5. При неудобных вопросах — уходишь от темы с ехидством."""


if __name__ == "__main__":
    main()
