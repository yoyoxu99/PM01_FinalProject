import re
import csv
import glob
import pandas as pd
from pathlib import Path


OUT = "summary_final.csv"
FR_AVG_OUT = "fr_avg.csv"


# =========================
# Parse filename
# =========================

def parse_name(path):

    name = Path(path).stem
    parts = name.split("_")

    method = parts[0].upper()

    prompt = "base"
    model = "unknown"
    run = "00"

    if "cot" in parts:
        prompt = "cot"

    if "mistral" in parts:
        model = "mistral"

    if parts[-1].isdigit():
        run = parts[-1]

    return method, prompt, model, run


# =========================
# Normalize ability
# =========================

def norm_ability(text):

    if text == "Overall":
        return "Overall"

    if "Second" in text:
        return "SecondOrder"

    return "FirstOrder"


# =========================
# FR parser
# =========================

def parse_fr(text):

    rows = []

    total = re.search(r"Total Exact Match Accuracy:\s+([0-9.]+)", text)

    if total:
        rows.append(("EM", "Overall", float(total.group(1)), None, None))


    for line in text.splitlines():

        m = re.search(r"(Belief:.*)\(n=.*?\):\s+([0-9.]+)", line)

        if not m:
            continue

        ab = norm_ability(m.group(1))
        acc = float(m.group(2))

        rows.append(("EM", ab, acc, None, None))


    return rows


# =========================
# MC parser
# =========================

def parse_mc(text):

    rows = []

    raw = re.search(r"Raw Acc\s+:\s+([0-9.]+)", text)
    vote = re.search(r"Vote Acc\s+:\s+([0-9.]+)", text)

    if raw:
        rows.append(("Raw", "Overall", float(raw.group(1)), None, None))

    if vote:
        rows.append(("Vote", "Overall", float(vote.group(1)), None, None))


    for setting, pattern in [

        ("Raw",
         r"Accuracy by ABILITY \(Raw\):([\s\S]*?)Accuracy by ABILITY"),

        ("Vote",
         r"Accuracy by ABILITY \(Vote\):([\s\S]*?)-----")
    ]:

        block = re.findall(pattern, text)

        if not block:
            continue


        for line in block[0].splitlines():

            m = re.search(r"(Belief:.*)\(n=.*?\):\s+([0-9.]+)", line)

            if not m:
                continue

            ab = norm_ability(m.group(1))
            acc = float(m.group(2))

            rows.append((setting, ab, acc, None, None))


    return rows


# =========================
# LM parser (with Margin)
# =========================

def parse_lm(text):

    rows = []


    # ---- Accuracy (Overall) ----

    raw = re.search(r"Total Accuracy \(Raw\):\s+([0-9.]+)", text)
    norm = re.search(r"Total Accuracy \(Normalized\):\s+([0-9.]+)", text)

    if raw:
        rows.append(("Raw", "Overall", float(raw.group(1)), "Raw", None))

    if norm:
        rows.append(("Norm", "Overall", float(norm.group(1)), "Norm", None))


    # ---- Accuracy by Ability ----

    for setting, pattern in [

        ("Raw",
         r"Accuracy by Ability \(Raw\):([\s\S]*?)Accuracy by Ability"),

        ("Norm",
         r"Accuracy by Ability \(Normalized\):([\s\S]*?)Total Margin")
    ]:

        block = re.findall(pattern, text)

        if not block:
            continue


        for line in block[0].splitlines():

            m = re.search(r"(Belief:.*)\(n=.*?\):\s+([0-9.]+)", line)

            if not m:
                continue

            ab = norm_ability(m.group(1))
            acc = float(m.group(2))

            rows.append((setting, ab, acc, setting, None))


    # ---- Margins ----

    margin_map = {}

    m_raw = re.search(r"Total Margin \(Raw\):\s+([-0-9.]+)", text)
    m_norm = re.search(r"Total Margin \(Normalized\):\s+([-0-9.]+)", text)

    if m_raw:
        margin_map["Raw_Overall"] = float(m_raw.group(1))

    if m_norm:
        margin_map["Norm_Overall"] = float(m_norm.group(1))


    for setting, pattern in [

        ("Raw",
         r"Margin by Ability \(Raw\):([\s\S]*?)Margin by Ability"),

        ("Norm",
         r"Margin by Ability \(Normalized\):([\s\S]*)$")
    ]:

        block = re.findall(pattern, text)

        if not block:
            continue


        for line in block[0].splitlines():

            m = re.search(r"(Belief:.*)\(n=.*?\):\s+([-0-9.]+)", line)

            if not m:
                continue

            ab = norm_ability(m.group(1))
            margin = float(m.group(2))

            margin_map[f"{setting}_{ab}"] = margin


    final = []

    for setting, ability, acc, mtype, _ in rows:

        key = f"{setting}_{ability}"

        margin = margin_map.get(key)

        final.append(
            (setting, ability, acc, mtype, margin)
        )


    return final


# =========================
# Main
# =========================

def main():

    files = glob.glob("logs/*.txt")

    if not files:
        print("No files found in logs/")
        return


    rows = []


    for f in files:

        text = Path(f).read_text(encoding="utf-8")

        method, prompt, model, run = parse_name(f)


        if "Free Response" in text:
            data = parse_fr(text)

        elif "MC Evaluation" in text:
            data = parse_mc(text)

        elif "LM Probing" in text:
            data = parse_lm(text)

        else:
            print(f"Unknown format: {f}")
            continue


        for setting, ability, acc, mtype, margin in data:

            rows.append([
                method,
                prompt,
                model,
                setting,
                ability,
                acc,
                run,
                mtype,
                margin
            ])


    # ---- Build DataFrame ----

    df = pd.DataFrame(rows, columns=[
        "Method",
        "Prompt",
        "Model",
        "Setting",
        "Ability",
        "Accuracy",
        "Run",
        "MarginType",
        "Margin"
    ])


    # ---- Sorting rules ----

    df["Method"] = pd.Categorical(
        df["Method"],
        ["LM", "MC", "FR"],
        ordered=True
    )

    df["Prompt"] = pd.Categorical(
        df["Prompt"],
        ["base", "cot"],
        ordered=True
    )


    df["Setting"] = pd.Categorical(
        df["Setting"],
        ["Raw", "Norm", "Vote", "EM"],
        ordered=True
    )


    df["Ability"] = pd.Categorical(
        df["Ability"],
        ["Overall", "FirstOrder", "SecondOrder"],
        ordered=True
    )


    df["Run"] = pd.to_numeric(
        df["Run"],
        errors="coerce"
    ).fillna(0).astype(int)


    df = df.sort_values(
        ["Method", "Prompt", "Setting", "Ability", "Run"]
    )


    # ---- Compute FR averages ----

    fr = df[
        (df["Method"] == "FR") &
        (df["Setting"] == "EM")
    ]


    results = []


    for prompt in ["base", "cot"]:

        for ability, tag in [
            ("Overall", "OR"),
            ("FirstOrder", "FO"),
            ("SecondOrder", "SO")
        ]:

            sub = fr[
                (fr["Prompt"] == prompt) &
                (fr["Ability"] == ability)
            ]

            avg = sub["Accuracy"].mean() if len(sub) else None

            name = f"FR_{prompt.upper()}_{tag}_AvgAcc"

            results.append([name, round(avg, 4)])


    # Save FR averages
    with open(FR_AVG_OUT, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        for r in results:
            writer.writerow(r)


    # ---- Save main table ----

    df.to_csv(OUT, index=False)

    print(f"Saved {len(df)} rows → {OUT}")
    print(f"Saved FR averages → {FR_AVG_OUT}")



if __name__ == "__main__":
    main()
