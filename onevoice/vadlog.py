"""Làm gọn log của tầng VAD/Segmenter.

`VAD-Segmenter/` là bàn giao của thành viên khác và quy ước repo là **không
sửa** (xem CLAUDE.md). Nhưng nó in rất dày:

* `VAD_process` in một dòng visualizer **mỗi frame 32ms** (~31 dòng/giây);
* `Segmenter_process` in 4-5 dòng mỗi lần chốt câu.

Chạy thật thì phụ đề bị đẩy trôi khỏi màn hình trước khi kịp đọc. Thay vì sửa
file của người khác, ở đây làm hai việc từ phía `onevoice/`:

1. `FilteredStdout` — tạm thay `sys.stdout` trong lúc VAD chạy và **nuốt đúng
   những dòng ồn đã biết**, mọi dòng khác (lỗi luồng âm thanh, cảnh báo
   overflow, thông báo dừng) vẫn cho qua. Không nuốt sạch: mất cảnh báo mic là
   mất manh mối gỡ lỗi quan trọng nhất của thiết bị này.
2. `VadTurnLogger` — gộp chuỗi frame VAD thành **lượt nói**, chỉ in 2 mốc:
   lúc bắt đầu có tiếng và lúc im đủ lâu để coi là hết lượt.

Lưu ý: `ConsoleDisplay` giữ tham chiếu tới stdout thật ngay từ lúc khởi tạo, nên
việc thay `sys.stdout` ở đây không ảnh hưởng phụ đề.
"""

import re
import sys

__all__ = ["FilteredStdout", "VadTurnLogger", "muted_stdout"]

# Frame VAD: 512 mẫu @ 16 kHz.
FRAME_SECONDS = 512.0 / 16000.0

# Tiền tố của những dòng cần nuốt. Khớp theo phần đầu dòng sau khi strip.
_NOISY_PREFIXES = (
    "🗣️",                      # visualizer "VAD: ĐANG NÓI"
    "🤫",                      # visualizer "VAD: KHOẢNG LẶNG"
    "🟢 [Segmenter]",          # "Bắt đầu câu mới"
    "🔴 [Segmenter]",          # "Chốt câu #..."
    "⏱",                      # "Thời lượng : ...s | Samples: ..."
    "📡",                      # "Đã gửi RAM : on_segment_ready() ..."
    "💾",                      # "Đã lưu WAV : file:///..."
    "📦",                      # "Đã nạp mô hình Silero VAD" — thay bằng dòng của ta
    "🎙️",                      # "Đã kết nối Micro [...]"  — thay bằng dòng của ta
    "🎛️",                      # "Tần số lấy mẫu phần cứng"
    "🚀 Đang chạy VAD Engine",
    "Hãy nói Tiếng Việt hoặc Tiếng Hoa",
    "----------",
)

_LINE_SPLIT = re.compile(r"(\r\n|\n|\r)")


class FilteredStdout(object):
    """Proxy stdout: nuốt dòng ồn của VAD/Segmenter, cho mọi dòng khác đi qua.

    Gom theo dòng vì `VAD_process` ghi visualizer bằng `\\r` không kèm `\\n` —
    lọc theo từng lần `write()` sẽ không bắt được trọn dòng.
    """

    def __init__(self, target=None, prefixes=_NOISY_PREFIXES):
        self._target = target if target is not None else sys.stdout
        self._prefixes = tuple(prefixes)
        self._buf = ""

    # ── Lọc ─────────────────────────────────────────────────────────────────
    def _is_noisy(self, line):
        stripped = line.strip()
        if not stripped:
            return True                     # dòng trống của Segmenter
        return stripped.startswith(self._prefixes)

    def write(self, text):
        if not text:
            return 0
        self._buf += text
        parts = _LINE_SPLIT.split(self._buf)
        self._buf = parts[-1]               # phần chưa có dấu xuống dòng
        for i in range(0, len(parts) - 1, 2):
            line = parts[i]
            if not self._is_noisy(line):
                # `\r` của visualizer đổi thành `\n`: dòng giữ lại là dòng thật,
                # không phải khung hình chồng lên nhau.
                self._target.write(line + "\n")
                self._target.flush()
        return len(text)

    def flush(self):
        # Cố tình KHÔNG xả `self._buf`: phần đuôi chưa xuống dòng gần như luôn là
        # khung visualizer đang vẽ dở. Xả ra là hỏng công lọc.
        self._target.flush()

    # ── Cho giống file object đủ để thư viện khác không vỡ ──────────────────
    def isatty(self):
        return getattr(self._target, "isatty", lambda: False)()

    def fileno(self):
        return self._target.fileno()

    def writable(self):
        return True

    def reconfigure(self, **kwargs):
        target_reconfigure = getattr(self._target, "reconfigure", None)
        if target_reconfigure is not None:
            target_reconfigure(**kwargs)

    @property
    def encoding(self):
        return getattr(self._target, "encoding", "utf-8")


class muted_stdout(object):
    """Context manager: thay `sys.stdout` bằng `FilteredStdout` rồi trả lại.

        with muted_stdout(enabled=not args.verbose):
            vad.start()

    `enabled=False` thì không làm gì — dùng cho `--verbose` khi cần xem lại log
    gốc của VAD/Segmenter lúc gỡ lỗi phần cứng.
    """

    def __init__(self, enabled=True, target=None):
        self.enabled = enabled
        self._target = target
        self._saved = None

    def __enter__(self):
        if self.enabled:
            self._saved = sys.stdout
            sys.stdout = FilteredStdout(self._target or self._saved)
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            sys.stdout = self._saved
            self._saved = None
        return False


class VadTurnLogger(object):
    """Gộp frame VAD thành **lượt nói** và in 2 mốc mỗi lượt.

        🎙️  ▶ Lượt 1 — có tiếng nói
        🔇  ■ Lượt 1 — im lặng, hết lượt (2.4s)

    Một lượt nói có thể bị Segmenter cắt thành **nhiều câu** (Energy Valley,
    Max Duration 8s), nên số lượt và số câu `#NNN` cố tình đánh riêng — thấy
    "lượt 3" ra hai câu là đúng, không phải lỗi.

    :param hangover_s: im bao lâu thì coi là hết lượt. Để bằng
        `silence_timeout` của Segmenter (0.6s) nên mốc "hết lượt" trùng đúng lúc
        Segmenter chốt câu — người dùng đọc log thấy nhân quả liền mạch.
    :param min_speech_s: cần bấy nhiêu giây có tiếng liên tiếp mới tính là mở
        lượt. Lọc frame nhiễu lẻ (tiếng gõ phím, cửa đóng) khỏi làm rác log.
    """

    def __init__(self, display, hangover_s=0.6, min_speech_s=0.1,
                 frame_s=FRAME_SECONDS, enabled=True):
        self.display = display
        self.enabled = enabled
        self.frame_s = frame_s
        self._hangover_frames = max(1, int(round(hangover_s / frame_s)))
        self._min_speech_frames = max(1, int(round(min_speech_s / frame_s)))

        self.turns = 0
        self._in_turn = False
        self._speech_run = 0        # số frame có tiếng liên tiếp (lúc chưa mở lượt)
        self._silence_run = 0       # số frame im liên tiếp (lúc đang trong lượt)
        self._turn_frames = 0

    def wrap(self, downstream):
        """Bọc `segmenter.process_vad_frame` — vẫn gọi tiếp nguyên vẹn.

        Chạy TRONG luồng VAD nên chỉ được đếm và in, tuyệt đối không xử lý nặng
        (xem docstring `onevoice/pipeline.py`).
        """
        def on_vad_frame(chunk, is_speech, prob, volume):
            self.feed(is_speech)
            return downstream(chunk, is_speech, prob, volume)
        return on_vad_frame

    def feed(self, is_speech):
        if not self._in_turn:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self._min_speech_frames:
                    self._open_turn()
            else:
                self._speech_run = 0
            return

        self._turn_frames += 1
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._hangover_frames:
                self._close_turn()

    # ── Mốc ─────────────────────────────────────────────────────────────────
    def _open_turn(self):
        self.turns += 1
        self._in_turn = True
        self._silence_run = 0
        # Tính cả số frame đã dùng để xác nhận mở lượt, nếu không độ dài lượt
        # luôn thiếu mất đoạn đầu.
        self._turn_frames = self._speech_run
        self._speech_run = 0
        self._emit("\n🎙️  ▶ Lượt %d — có tiếng nói" % self.turns)

    def _close_turn(self):
        # Trừ phần im lặng đuôi: đó là khoảng lặng chốt câu, không phải tiếng nói.
        spoken = max(0.0, (self._turn_frames - self._silence_run) * self.frame_s)
        self._emit("🔇  ■ Lượt %d — im lặng, hết lượt (%.1fs nói)"
                   % (self.turns, spoken))
        self._in_turn = False
        self._turn_frames = 0
        self._silence_run = 0
        self._speech_run = 0

    def _emit(self, message):
        if self.enabled:
            self.display.show_status(message)
