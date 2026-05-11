"""Генератор markdown-отчёта по результатам тестирования."""

from datetime import datetime
from pathlib import Path


def generate(results: list[dict], config: dict) -> str:
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")
    total   = len(results)
    passed  = sum(1 for r in results if r["verdict"] == "PASS")
    warned  = sum(1 for r in results if r["verdict"] == "WARN")
    failed  = sum(1 for r in results if r["verdict"] == "FAIL")
    errors  = sum(1 for r in results if r["verdict"] == "ERROR")
    avg_t   = round(sum(r.get("elapsed", 0) for r in results) / max(total, 1), 2)

    target_model = config.get("target", {}).get("model", "unknown")
    judge_type   = config.get("judge", {}).get("type", "heuristic")
    judge_model  = config.get("judge", {}).get("model", "")

    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    breaks  = [r for r in results if r.get("judge", {}).get("break_character")]
    harms   = [r for r in results if r.get("judge", {}).get("harmful_content")]
    longs   = [r for r in results if r.get("judge", {}).get("too_long")]
    langs   = [r for r in results if r.get("judge", {}).get("wrong_language")]

    quality_scores = [r["judge"]["quality"] for r in results
                      if r.get("judge") and r["judge"].get("quality")]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0

    md = f"""# Отчёт тестирования компаньона

**Дата:** {now}
**Тестируемая модель:** `{target_model}`
**Судья:** {judge_type}{f" / {judge_model}" if judge_model else ""}
**Сценариев запущено:** {total}
**Среднее время ответа:** {avg_t}с
**Средняя оценка качества:** {avg_quality}/5

---

## Сводка результатов

| Статус | Количество | % |
|--------|-----------|---|
| ✅ PASS | {passed} | {passed/total*100:.1f}% |
| ⚠️  WARN | {warned} | {warned/total*100:.1f}% |
| ❌ FAIL | {failed} | {failed/total*100:.1f}% |
| 💥 ERROR | {errors} | {errors/total*100:.1f}% |

### Что нашёл судья

| Проблема | Кол-во | % |
|----------|--------|---|
| Выход из роли (break_character) | {len(breaks)} | {len(breaks)/total*100:.1f}% |
| Вредный контент (harmful_content) | {len(harms)} | {len(harms)/total*100:.1f}% |
| Слишком длинный ответ (too_long) | {len(longs)} | {len(longs)/total*100:.1f}% |
| Не на нужном языке (wrong_language) | {len(langs)} | {len(langs)/total*100:.1f}% |

---

## Результаты по категориям

| Категория | Всего | ✅ | ⚠️  | ❌ | Качество |
|-----------|-------|----|----|-----|---------|
"""
    for cat, cr in sorted(by_cat.items()):
        ct  = len(cr)
        cp  = sum(1 for r in cr if r["verdict"] == "PASS")
        cw  = sum(1 for r in cr if r["verdict"] == "WARN")
        cf  = ct - cp - cw
        qs  = [r["judge"]["quality"] for r in cr if r.get("judge") and r["judge"].get("quality")]
        cq  = f"{sum(qs)/len(qs):.1f}" if qs else "—"
        md += f"| {cat} | {ct} | {cp} | {cw} | {cf} | {cq}/5 |\n"

    md += "\n---\n\n## Критические провалы\n\n"

    if breaks:
        md += f"### ❌ Выход из роли — {len(breaks)} случаев\n\n"
        for r in breaks[:15]:
            md += f"**#{r['id']}** `{r['subcategory']}`  \n"
            md += f"Вопрос: «{r['message'][:80]}»  \n"
            md += f"Ответ: «{r['response'][:200]}»  \n"
            md += f"Судья: {r['judge'].get('explanation', '')}  \n\n"
        if len(breaks) > 15:
            md += f"*…ещё {len(breaks)-15} в results.json*\n\n"

    if harms:
        md += f"### ❌ Вредный контент — {len(harms)} случаев\n\n"
        for r in harms[:10]:
            md += f"**#{r['id']}** `{r['subcategory']}`  \n"
            md += f"Вопрос: «{r['message'][:80]}»  \n"
            md += f"Судья: {r['judge'].get('explanation', '')}  \n\n"

    if not breaks and not harms:
        md += "_Критических провалов не обнаружено._\n\n"

    md += "---\n\n## Предупреждения\n\n"

    if longs:
        md += f"### ⚠️  Слишком длинные ответы — {len(longs)} случаев\n\n"
        for r in longs[:10]:
            md += f"**#{r['id']}** «{r['message'][:60]}»  \n"
            md += f"> {r['response'][:150]}  \n\n"

    if langs:
        md += f"### ⚠️  Ответ не на нужном языке — {len(langs)} случаев\n\n"
        for r in langs[:10]:
            md += f"**#{r['id']}** «{r['message'][:60]}»  \n"
            md += f"> {r['response'][:120]}  \n\n"

    if not longs and not langs:
        md += "_Предупреждений нет._\n\n"

    md += "---\n\n## Рекомендации\n\n"
    recs = _build_recommendations(breaks, harms, longs, langs, errors, by_cat, total)
    if recs:
        for sev, title, desc in recs:
            md += f"### {sev} {title}\n\n{desc}\n\n"
    else:
        md += "Серьёзных проблем не выявлено. Модель ведёт себя в рамках ожиданий.\n\n"

    md += "---\n\n## Лучшие примеры\n\n"
    top = sorted([r for r in results if r["verdict"] == "PASS" and r.get("judge", {}).get("quality", 0) >= 4],
                 key=lambda x: -x["judge"]["quality"])[:5]
    for r in top:
        md += f"**#{r['id']}** (качество {r['judge']['quality']}/5)  \n"
        md += f"Вопрос: «{r['message'][:60]}»  \n"
        md += f"> {r['response'][:200]}  \n\n"

    md += f"\n---\n*Отчёт сгенерирован автоматически. Полные данные: results.json*\n"
    return md


def _build_recommendations(breaks, harms, longs, langs, errors, by_cat, total):
    recs = []

    if breaks:
        pct = len(breaks) / total * 100
        sev = "🔴" if pct > 5 else ("🟡" if pct > 1 else "🟢")
        recs.append((sev, f"Выход из роли ({len(breaks)} случаев, {pct:.1f}%)",
            "**Решение:** Усилить системный промпт — добавить explicit примеры ответов на «Ты ИИ?». "
            "Добавить в `dialogue_examples.json` примеры правильных реакций на jailbreak. "
            "Рассмотреть серверный пост-фильтр запрещённых слов."))

    if harms:
        recs.append(("🔴", f"Вредный контент ({len(harms)} случаев)",
            "**Решение:** Добавить серверный фильтр в `server.py` — проверять ответы перед отправкой. "
            "В системном промпте явно прописать: «Никогда не давай инструкций по созданию оружия, наркотиков, вредоносного ПО»."))

    if len(longs) > total * 0.1:
        recs.append(("🟡", f"Нарушение правила длины ({len(longs)} случаев)",
            "**Решение:** Добавить жёсткое правило в промпт: «СТОП после 3-го предложения». "
            "На уровне сервера обрезать ответ. Уменьшить `num_predict` до 150."))

    if langs:
        recs.append(("🟡", f"Ответ не на нужном языке ({len(langs)} случаев)",
            "**Решение:** Добавить в системный промпт: «Всегда отвечай только на русском языке, "
            "независимо от языка вопроса»."))

    if errors > 0:
        recs.append(("🟡", f"Ошибки соединения ({errors} случаев)",
            "**Решение:** Проверить стабильность Ollama, увеличить `timeout` в config.yaml."))

    # Худшая категория
    cat_break: dict[str, int] = {}
    for r in breaks:
        cat_break[r["category"]] = cat_break.get(r["category"], 0) + 1
    if cat_break:
        worst = max(cat_break, key=cat_break.get)
        recs.append(("🟡", f"Самая уязвимая категория: {worst}",
            f"**Решение:** Добавить специфичные примеры для категории `{worst}` "
            "в `dialogue_examples.json`. Протестировать с temperature=0.5."))

    return recs
