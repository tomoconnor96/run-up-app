def build_report_html(scores, notes, phases, fps, video_name):
    order = ["RUN-UP", "GATHER", "BACK FOOT CONTACT", "FRONT FOOT CONTACT", "RELEASE / SNAP", "FOLLOW-THROUGH"]
    phase_time = {}
    for name, a, b in phases:
        phase_time[name] = a / fps

    rows = ""
    total, count = 0, 0
    for name in order:
        score = scores.get(name)
        note = notes.get(name, "")
        t = phase_time.get(name, 0)
        if score is None:
            score_html = '<span class="score na">--</span>'
        else:
            total += score
            count += 1
            score_html = f'<span class="score s{score}">{score}/10</span>'
        rows += f"""
        <tr>
          <td class="phase">{name}</td>
          <td class="time">{t:.2f}s</td>
          <td>{score_html}</td>
          <td class="note">{note}</td>
        </tr>"""

    overall = round(total / count, 1) if count else 0

    return rows, overall
