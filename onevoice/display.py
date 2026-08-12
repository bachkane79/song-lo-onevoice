"""Tầng hiển thị — bước cuối của pipeline.

Mặc định in ra terminal. Khi gắn màn hình thật (LCD SPI, HDMI, web UI...) chỉ cần
viết một class mới kế thừa `Display` và cắm vào pipeline — không phải sửa gì ở
các tầng trước.

    class LcdDisplay(Display):
        def show(self, event):
            my_screen.draw(event.source_text, event.target_text)
"""

import json
import sys
import time
from dataclasses import dataclass, field

__all__ = ["DisplayEvent", "Display", "ConsoleDisplay", "JsonlDisplay",
           "CompositeDisplay"]

# Nhãn hiển thị theo mã ngôn ngữ ngắn
_LANG_LABEL = {"zh": "中文", "vi": "Tiếng Việt", None: "?"}
_FLAG = {"zh": "🇨🇳", "vi": "🇻🇳", None: "❓"}


@dataclass
class DisplayEvent:
    """Một câu đã đi hết pipeline, sẵn sàng hiển thị."""

    seq: int                       # số thứ tự câu
    source_text: str               # câu STT nghe được
    target_text: str               # bản dịch
    src_lang: str = None           # "zh" | "vi"
    tgt_lang: str = None
    audio_duration_s: float = 0.0  # độ dài câu nói
    stt_ms: float = 0.0
    mt_ms: float = 0.0
    total_ms: float = 0.0          # từ lúc Segmenter chốt câu đến lúc hiển thị
    warnings: tuple = field(default_factory=tuple)   # ví dụ: có ký tự <unk>
    # Chi tiết định tuyến ngôn ngữ của STT song ngữ (chỉ để gỡ lỗi/hiển thị)
    stt_detail: dict = None

    @property
    def lag_ratio(self):
        """total_ms / độ dài câu nói.

        < 1.0 nghĩa là xử lý nhanh hơn tốc độ nói → phụ đề bám kịp hội thoại.
        > 1.0 nghĩa là mỗi câu lại tụt lại thêm một chút, hàng đợi sẽ dồn.
        """
        if self.audio_duration_s <= 0:
            return 0.0
        return self.total_ms / (self.audio_duration_s * 1000.0)


class Display(object):
    """Interface hiển thị."""

    def show(self, event):
        """Hiển thị một câu đã dịch xong."""
        raise NotImplementedError

    def show_status(self, message):
        """Thông báo trạng thái (khởi động, cảnh báo, dừng)."""

    def close(self):
        """Đóng màn hình / file."""


class ConsoleDisplay(Display):
    """In ra terminal — mặc định khi chưa gắn màn hình thật.

    Bố cục mỗi câu (nguồn ở trên, bản dịch nổi bật ở dưới)::

        ──────────────────────────────────────────────
        #003  🇨🇳 中文  ·  2.4s nói  ·  0.7s xử lý
          你好，请问厕所在哪里？
        → Xin chào, xin hỏi nhà vệ sinh ở đâu?
    """

    # Mã ANSI; tự tắt khi output bị pipe vào file (không phải terminal)
    _DIM, _BOLD, _CYAN, _YELLOW, _RESET = (
        "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[0m")

    def __init__(self, stream=None, color=None, show_timing=True, width=62,
                 show_route=False):
        self.stream = stream or sys.stdout
        if color is None:
            color = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.color = color
        self.show_timing = show_timing
        self.show_route = show_route
        self.width = width
        if hasattr(self.stream, "reconfigure"):
            try:
                self.stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, AttributeError):
                pass

    def _c(self, code, text):
        return "%s%s%s" % (code, text, self._RESET) if self.color else text

    def show(self, event):
        head = "#%03d  %s %s" % (event.seq, _FLAG.get(event.src_lang, "❓"),
                                 _LANG_LABEL.get(event.src_lang, "?"))
        if self.show_timing:
            head += "  ·  %.1fs nói  ·  %.2fs xử lý (STT %.0fms + dịch %.0fms)" % (
                event.audio_duration_s, event.total_ms / 1000.0,
                event.stt_ms, event.mt_ms)
            if event.lag_ratio > 1.0:
                head += "  ⚠️ chậm hơn tốc độ nói"

        out = self.stream
        out.write("\n" + self._c(self._DIM, "─" * self.width) + "\n")
        out.write(self._c(self._DIM, head) + "\n")
        out.write("  " + self._c(self._CYAN, event.source_text) + "\n")
        out.write(self._c(self._BOLD, "→ " + event.target_text) + "\n")
        if self.show_route and event.stt_detail:
            d = event.stt_detail
            out.write(self._c(self._DIM,
                              "  ↳ định tuyến %s | vi_rate=%s zh_rate=%s\n"
                              "    zh nghe: %s\n    vi nghe: %s"
                              % (d.get("route"), d.get("vi_rate"), d.get("zh_rate"),
                                 (d.get("zh_text") or "")[:52],
                                 (d.get("vi_text") or "")[:52])) + "\n")
        for w in event.warnings:
            out.write(self._c(self._YELLOW, "  ⚠️  " + w) + "\n")
        out.flush()

    def show_status(self, message):
        self.stream.write(self._c(self._DIM, message) + "\n")
        self.stream.flush()


class JsonlDisplay(Display):
    """Ghi mỗi câu thành một dòng JSON — để log lại phiên chạy mà phân tích sau.

    Ghi ra file, KHÔNG ghi ra thẻ nhớ liên tục khi chạy production trên thiết bị
    (mòn thẻ). Chỉ bật khi đang thu thập dữ liệu đánh giá.
    """

    def __init__(self, path):
        self.path = path
        self._f = open(path, "a", encoding="utf-8")

    def show(self, event):
        row = dict(event.__dict__)
        row["warnings"] = list(row.get("warnings", ()))
        row["ts"] = time.time()
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self):
        if not self._f.closed:
            self._f.close()


class CompositeDisplay(Display):
    """Gửi cùng một sự kiện tới nhiều màn hình (vd: terminal + file log)."""

    def __init__(self, *displays):
        self.displays = [d for d in displays if d is not None]

    def show(self, event):
        for d in self.displays:
            d.show(event)

    def show_status(self, message):
        for d in self.displays:
            d.show_status(message)

    def close(self):
        for d in self.displays:
            d.close()
