"""Tự chọn micro — để không phải gõ `--device` mỗi lần chạy.

`SileroVADDetector` khi nhận `device_id=None` chỉ lấy `sd.default.device[0]` rồi
dùng luôn, không thử mở. Trên Windows nhiều card âm thanh, mục "default" đó có
thể là thiết bị **không mở được** (Sound Mapper trỏ tới card đã rút, Stereo Mix
đang tắt, mic bị app khác chiếm) — và lỗi chỉ lộ ra khi `InputStream` đã chạy,
lúc đó pipeline đã nạp xong 655 MB model.

Module này **mở thử từng ứng viên** rồi mới trả ID về. Đã trả về là chắc chắn
thu được. Thứ tự thử:

    1. Thiết bị người dùng chỉ định (`--device N`) — sai thì báo lỗi, không lặng
       lẽ đổi sang cái khác.
    2. Thiết bị mặc định của hệ điều hành — tôn trọng lựa chọn của người dùng.
    3. Các mic thật còn lại (tên có "microphone"/"headset"...).
    4. Cuối cùng mới tới các cổng ảo (Stereo Mix, Sound Mapper, loopback) — thu
       được tiếng loa chứ không phải tiếng người nói, chọn nhầm thì phụ đề sẽ
       dịch chính bản dịch vừa phát ra.
"""

__all__ = ["input_devices", "probe_device", "pick_input_device",
           "describe_device", "MicNotFound"]

# Cổng ảo / loopback — vẫn thử được nhưng để cuối hàng.
_DEPRIORITIZE = (
    "stereo mix", "sound mapper", "primary sound capture", "pc speaker",
    "loopback", "what u hear", "wave out mix", "mixage", "output",
)
# Dấu hiệu của mic thật.
_PREFER = ("microphone", "mic ", "mic(", "headset", "array", "webcam", "line in")


class MicNotFound(RuntimeError):
    """Không mở được micro nào — kèm danh sách đã thử để người dùng tự chọn."""


def _sd():
    import sounddevice as sd
    return sd


def input_devices():
    """Danh sách ``(index, info)`` của mọi thiết bị thu được âm."""
    return [(idx, dev) for idx, dev in enumerate(_sd().query_devices())
            if dev.get("max_input_channels", 0) > 0]


def describe_device(device_id):
    """Chuỗi mô tả ngắn: ``[1] Microphone Array (4 kênh @ 44100 Hz)``."""
    info = _sd().query_devices(device_id, "input")
    return "[%d] %s (%d kênh @ %.0f Hz)" % (
        device_id, info["name"], info["max_input_channels"],
        info["default_samplerate"])


def probe_device(device_id):
    """Mở thử `InputStream` đúng như VAD sẽ mở. True = dùng được thật.

    Mở rồi đóng ngay, không đọc frame nào: cái cần bắt là lỗi chiếm dụng /
    thiết bị không tồn tại, và lỗi đó xảy ra ngay lúc mở. Đọc thêm audio chỉ làm
    chậm lúc khởi động mà không biết thêm gì — mic im lặng vẫn là mic tốt.
    """
    sd = _sd()
    try:
        info = sd.query_devices(device_id, "input")
        channels = int(info["max_input_channels"])
        if channels <= 0:
            return False
        # Đúng tham số VAD_process._init_device() dùng: sample rate gốc của
        # thiết bị + toàn bộ kênh. Thử bằng tham số khác thì phép thử vô nghĩa.
        with sd.InputStream(device=device_id,
                            samplerate=int(info["default_samplerate"]),
                            channels=channels, dtype="float32"):
            pass
        return True
    except Exception:       # noqa: BLE001 — mọi lỗi đều nghĩa là "không dùng được"
        return False


def _rank(idx, dev, default_idx):
    """Điểm càng nhỏ càng được thử trước."""
    if idx == default_idx:
        return (-1, idx)
    name = (dev.get("name") or "").lower()
    if any(hint in name for hint in _DEPRIORITIZE):
        return (2, idx)
    if any(hint in name for hint in _PREFER):
        return (0, idx)
    return (1, idx)


def pick_input_device(preferred=None):
    """Trả về ``(device_id, tên)`` của micro dùng được.

    :param preferred: ID người dùng chỉ định. Nếu mở không được thì **báo lỗi**
        chứ không tự đổi — người dùng gõ `--device 5` là có ý, im lặng chạy
        device khác sẽ khiến họ tưởng mic đó hỏng.
    :raises MicNotFound: máy không có mic, hoặc không mic nào mở được.
    """
    sd = _sd()

    if preferred is not None:
        if not probe_device(preferred):
            raise MicNotFound(
                "Không mở được micro [%s].\n%s" % (preferred, _available_hint()))
        return preferred, sd.query_devices(preferred, "input")["name"]

    devices = input_devices()
    if not devices:
        raise MicNotFound("Máy không có thiết bị thu âm nào. Cắm micro rồi chạy lại.")

    try:
        default_idx = sd.default.device[0]
    except Exception:       # noqa: BLE001
        default_idx = None

    tried = []
    for idx, dev in sorted(devices, key=lambda p: _rank(p[0], p[1], default_idx)):
        if probe_device(idx):
            return idx, dev["name"]
        tried.append("[%d] %s" % (idx, dev["name"]))

    raise MicNotFound(
        "Có %d thiết bị thu nhưng không mở được cái nào.\n"
        "Đã thử: %s\n"
        "→ Kiểm tra Windows Settings → Privacy → Microphone, hoặc tắt app đang "
        "chiếm mic (Zoom/Teams/Discord)." % (len(devices), ", ".join(tried)))


def _available_hint():
    lines = ["Micro đang có trên máy:"]
    for idx, dev in input_devices():
        mark = "✓" if probe_device(idx) else "✗"
        lines.append("   %s [%2d] %s" % (mark, idx, dev["name"]))
    lines.append("→ Bỏ --device để pipeline tự chọn.")
    return "\n".join(lines)
