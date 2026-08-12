"""Đo máy dịch trên máy thật — chạy cái này TRƯỚC khi chốt cấu hình Jetson.

Mọi con số hiệu năng trong tài liệu repo này đều đo trên máy dev x86 (i9-9900K).
CPU Jetson Nano là 4×Cortex-A57 1.43 GHz, chậm hơn nhiều lần và **không suy ra
được bằng phép nhân**. Đừng tin số ngoại suy — chạy script này trên đúng thiết bị
sẽ ship.

    python3 scripts/bench_translate.py                  # cấu hình mặc định
    python3 scripts/bench_translate.py --sweep-beam     # so beam 1/2/4
    python3 scripts/bench_translate.py --sweep-threads  # so 1/2/4 thread
    python3 scripts/bench_translate.py --json out.json

Đọc kết quả:
  - `ms/câu` là ngân sách dịch. Cộng thêm STT rồi so với độ dài câu nói (~2-4s).
    Tổng vượt quá độ dài câu nói nghĩa là hàng đợi sẽ dồn và pipeline bắt đầu bỏ câu.
  - `peak RSS` phải còn chỗ cho STT + hệ điều hành. Jetson Nano 4 GB thì thoải
    mái; bản 2 GB thì phải tính kỹ.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onevoice.config import TranslateConfig  # noqa: E402

# Câu hội thoại du lịch thật, độ dài trải từ rất ngắn đến dài —
# đúng loại câu thiết bị gặp, không phải câu tin tức dài như bộ benchmark gốc.
BENCH_ZH = [
    "太贵了。",
    "这个多少钱？",
    "你好，请问卫生间在哪里？",
    "我想订一间双人房，住两个晚上。",
    "这个菜里面有花生吗？我对花生过敏。",
    "请问从这里到机场坐出租车大概需要多长时间？",
    "我们明天早上八点在酒店大堂集合，然后一起去参观博物馆。",
]
BENCH_VI = [
    "Bao nhiêu tiền?",
    "Cho tôi xem thực đơn.",
    "Nhà vệ sinh ở đâu ạ?",
    "Tôi muốn đặt hai phòng đôi cho tối nay.",
    "Chuyến xe buýt cuối cùng về trung tâm khởi hành lúc mấy giờ?",
    "Anh cho tôi hỏi từ đây đi Hà Nội mất khoảng bao lâu bằng tàu hỏa?",
    "Chúng tôi sẽ ở lại đây ba ngày, sau đó bay vào Đà Nẵng rồi về nước.",
]


def peak_rss_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1024.0 ** 2
    except ImportError:
        try:                                    # Linux/Jetson: không cần psutil
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except (ImportError, AttributeError):
            return 0.0


def run_one(cfg, label, verbose=True):
    from onevoice.translate import NllbTranslator

    tr = NllbTranslator(cfg)
    tr.warmup()

    rows = []
    for lang, sentences in (("zh", BENCH_ZH), ("vi", BENCH_VI)):
        for text in sentences:
            t0 = time.perf_counter()
            res = tr.translate(text, lang)
            rows.append({
                "lang": lang, "src": text, "hyp": res.target,
                "ms": (time.perf_counter() - t0) * 1000.0,
                "out_tokens": res.out_tokens,
                "unk": list(res.unk_chars),
            })
            if verbose:
                print("   %s  %5.0f ms  %s" % (lang, rows[-1]["ms"], res.target))

    total_ms = sum(r["ms"] for r in rows)
    total_tok = sum(r["out_tokens"] for r in rows)
    result = {
        "label": label,
        "beam_size": cfg.beam_size,
        "intra_threads": cfg.intra_threads,
        "compute_type": cfg.compute_type,
        "load_s": round(tr.load_seconds, 2),
        "sentences": len(rows),
        "mean_ms": round(total_ms / len(rows), 1),
        "median_ms": round(sorted(r["ms"] for r in rows)[len(rows) // 2], 1),
        "max_ms": round(max(r["ms"] for r in rows), 1),
        "tokens_per_s": round(total_tok / (total_ms / 1000.0), 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rows": rows,
    }
    tr.close()
    return result


def print_table(results):
    print("\n%-22s %6s %8s %10s %10s %9s %10s"
          % ("cấu hình", "beam", "thread", "ms/câu TB", "ms/câu max", "tok/s", "peak RSS"))
    print("-" * 82)
    for r in results:
        print("%-22s %6d %8d %10.0f %10.0f %9.1f %9.0f M"
              % (r["label"], r["beam_size"], r["intra_threads"], r["mean_ms"],
                 r["max_ms"], r["tokens_per_s"], r["peak_rss_mb"]))
    print("""
Ngân sách thực tế: câu nói dài 2-4s. Dịch + STT phải gọn hơn con số đó, nếu
không hàng đợi dồn và pipeline sẽ bỏ câu (xem thống kê khi dừng run_pipeline.py).""")


def main():
    ap = argparse.ArgumentParser(description="Đo máy dịch trên thiết bị thật")
    ap.add_argument("--sweep-beam", action="store_true", help="So beam 1, 2, 4")
    ap.add_argument("--sweep-threads", action="store_true", help="So 1, 2, 4 thread")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--json", metavar="FILE", help="Ghi kết quả đầy đủ ra JSON")
    ap.add_argument("--quiet", action="store_true", help="Không in từng câu")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base = TranslateConfig()
    if args.beam:
        base.beam_size = args.beam
    if args.threads:
        base.intra_threads = args.threads

    configs = []
    if args.sweep_beam:
        for b in (1, 2, 4):
            c = TranslateConfig(**dict(base.__dict__))
            c.beam_size = b
            configs.append(("beam %d" % b, c))
    elif args.sweep_threads:
        for t in (1, 2, 4):
            c = TranslateConfig(**dict(base.__dict__))
            c.intra_threads = t
            configs.append(("%d thread" % t, c))
    else:
        configs.append(("mặc định", base))

    results = []
    for label, cfg in configs:
        print("\n=== %s (beam %d, %d thread) ==="
              % (label, cfg.beam_size, cfg.intra_threads))
        results.append(run_one(cfg, label, verbose=not args.quiet))

    print_table(results)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n✅ Đã ghi %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
