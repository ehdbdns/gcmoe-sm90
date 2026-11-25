"""Fill the measured section of the EP8 note from the full-model study CSVs."""
import csv
import math
from itertools import combinations
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
RESULTS = DOCS / "trace_executor_20261119"
BS256 = RESULTS / "sm90_bs256_main_five_plan_comm_results/delivery/bs256"
V2 = RESULTS / "sm90_bs512_bs1024_full_model_four_results"
REPLY = DOCS / "trace_handover_20261119/REPLY_SM90_BS512_BS1024_FULL_MODEL_FOUR.md"
ARMS = {"vanilla": "原生", "random": "随机 + RR", "rr": "EPLB + RR", "eco": "EPLB + ECO"}


def read_group(folder, name, bs):
    with (folder / f"{name}.csv").open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if int(row["bs"]) == bs]


def number(row, field):
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"invalid {field}: {value}")
    return value


def pct(a, b):
    return 100 * (a / b - 1)


def change(p):
    return "0.00%" if round(p, 2) == 0 else f"{p:+.2f}%"


def load(bs, folder):
    rows = [r for r in read_group(folder, "main_table", bs) if r["arm"] in ARMS]
    data = {r["arm"]: r for r in rows}
    pairs = [r for r in read_group(folder, "pairs", bs) if r["arm_a"] in ARMS and r["arm_b"] in ARMS]
    expected = {frozenset(pair) for pair in combinations(ARMS, 2)}
    if len(rows) != 4 or data.keys() != ARMS.keys() or len(pairs) != 6 or {
        frozenset((r["arm_a"], r["arm_b"])) for r in pairs
    } != expected:
        raise ValueError("四臂或六对对照不完整")
    for r in pairs:
        if (int(r["native_diff"]) or int(r["measured_samples"]) != bs * 1000
                or r["warm_native_equal"] != "True" or r["strict_pair_comparable"] != "True"):
            raise ValueError(f"{r['arm_a']} / {r['arm_b']} 未通过公平性核验")
    if any(r["primary_price"] != "fine" for r in rows) or int(data["eco"]["auction_slack"]) != 11469:
        raise ValueError("实测 cost 配置与实验设定不符")
    auction, = (r for r in read_group(folder, "auction", bs) if r["arm"] == "eco")
    if int(auction["calls"]) != 40000:
        raise ValueError("拍卖记录少于或超过 40,000 次层调用")
    times = {arm: number(row, "moe_ms_per_step") * 1000 / 40 for arm, row in data.items()}
    table = []
    for arm, label in ARMS.items():
        r = data[arm]
        fields = [str(bs), label, f"{times[arm]:.2f}", change(pct(times[arm], times["vanilla"])),
                  change(pct(times[arm], times["rr"]))]
        for field, digits in (("token_ir", 3), ("total_m_blocks", 2), ("peak_rank_m_blocks", 2),
                              ("active_physical_slots", 2), ("m_block_ir", 3),
                              ("routed_mae_pp", 4), ("graph_mae_pp", 4)):
            fields.append(f"{number(r, field):.{digits}f}")
        fields.append(f"{number(auction, 'accepted_per_layer_call'):.2f}" if arm == "eco" else "—")
        table.append("| " + " | ".join(fields) + " |")
    delta = times["eco"] - times["rr"]
    direction = "缩短" if delta < 0 else "增加" if delta > 0 else "持平"
    source = (folder / "main_table.csv").relative_to(DOCS)
    line = (f"- **BS{bs}**　EPLB 上从 RR 切到 ECO，MoE 从 {times['rr']:.2f} 到 "
            f"{times['eco']:.2f} µs/层，{direction} {abs(delta):.2f} µs/层"
            f"（{change(pct(times['eco'], times['rr']))}）。拍卖接受 {int(auction['accepted']):,} / "
            f"{int(auction['offers']):,} 次报价（{number(auction, 'acceptance_pct'):.2f}%），"
            f"零成交 {int(auction['calls_zero_accepts']):,} 次层调用。[原始 CSV]({source})")
    return line, table


def render():
    notes, rows = [], []
    for bs in (256, 512, 1024):
        folder = BS256 if bs == 256 else V2 / f"delivery-{bs}" / f"bs{bs}"
        if bs != 256 and not (folder / "main_table.csv").is_file():
            folder = V2  # The launcher also emits merged CSVs directly under the job.
        try:
            note, table = load(bs, folder)
        except FileNotFoundError:
            if bs == 256:
                raise
            if REPLY.is_file():
                source = REPLY.relative_to(DOCS)
                notes.append(f"- **BS{bs}**　归档中尚缺完整正式 CSV。[执行回执]({source})")
            else:
                notes.append(f"- **BS{bs}**　等待第二版整模结果。")
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            notes.append(f"- **BS{bs}**　归档数据未通过核验（{exc}），暂无法比较。")
        else:
            notes.append(note)
            rows.extend(table)
    return "\n".join([*notes, "", "## 整模主表", "",
        "| BS | 臂 | MoE（µs/层） | 相对原生 | 相对 RR | token IR | 总 M 块 | 峰值 rank M 块 | 活跃物理槽 | M 块 IR | routed MAE（百分点） | graph MAE（百分点） | 成交/层调用 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *rows, "",
        "每次层调用先取八个 rank 从 graph 起点至 combine 结束的最大跨度，再对 1000 步 × 40 层等权平均。变化为本臂耗时除以同档参考臂耗时再减一，负值表示更快。总 M 块与活跃槽按八卡求和，峰值 M 块按 rank 取最大；IR 为每次层调用的八卡最大值除以均值。MAE 比较各 rank 的预测与实测耗时份额。表中每档四臂的六对对照均满足预热输出一致、正式 top‑1 零差异。"])


if __name__ == "__main__":
    report = DOCS / "SM90_EP8_DECODE.md"
    before, body = report.read_text().split("<!-- ep8-results -->")
    _, after = body.split("<!-- /ep8-results -->")
    report.write_text(before + "<!-- ep8-results -->\n\n" + render() + "\n\n<!-- /ep8-results -->" + after)
