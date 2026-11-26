"""Zero-dependency Vietnamese word segmentation & compound phrase extraction.

Automatically uses `pyvi` or `underthesea` if installed in the environment, and always
falls back to a fast, deterministic maximal-forward-matching segmenter backed by a
curated technical/business/general Vietnamese compound lexicon so topic labels and
entity extraction never break Vietnamese words into meaningless single syllables.
"""

from functools import lru_cache
import re
import unicodedata

# Curated Vietnamese compound words (2–4 syllables) prioritized for maximal forward matching.
# Covers general, business, manufacturing/operations, and software engineering domains.
VI_COMPOUND_WORDS: frozenset[str] = frozenset({
    # 3-4 syllable compounds
    "cơ sở dữ liệu", "trí tuệ nhân tạo", "học máy sâu", "xử lý ngôn ngữ",
    "kho tri thức", "đồ thị tri thức", "bản đồ chủ đề", "dòng thời gian",
    "hệ điều hành", "kiến trúc hệ thống", "giao diện người dùng", "trải nghiệm người dùng",
    "tài liệu kỹ thuật", "quy trình làm việc", "quy trình vận hành", "quy trình sản xuất",
    "quy trình nạp nguyên liệu", "nạp nguyên liệu", "kiểm tra chất lượng", "quản lý chất lượng",
    "quản lý dự án", "quản lý nhân sự", "quản lý kho", "quản lý người dùng",
    "quản lý phiên", "bảo mật thông tin", "an toàn thông tin", "phân quyền truy cập",
    "xác thực người dùng", "mã nguồn mở", "môi trường sản xuất", "triển khai tự động",
    "kiểm thử tự động", "kiểm thử tích hợp", "hiệu năng hệ thống", "tối ưu hiệu năng",
    "khôi phục dữ liệu", "sao lưu dữ liệu", "đồng bộ dữ liệu", "lưu trữ dữ liệu",
    "truy xuất thông tin", "tìm kiếm ngữ nghĩa", "tìm kiếm toàn văn", "bộ nhớ dài hạn",
    "bộ nhớ ngắn hạn", "bộ nhớ đệm", "mô hình ngôn ngữ", "nhúng văn bản",
    "chế độ nhà phát triển", "cà phê sữa đá", "phiếu kiểm tra", "phiếu xuất kho",
    "phiếu nhập kho", "báo cáo tiến độ", "báo cáo tổng hợp", "thông số kỹ thuật",
    "hướng dẫn sử dụng", "hướng dẫn cài đặt", "tài khoản người dùng", "nhật ký hệ thống",
    # 2 syllable compounds — tech & engineering
    "hệ thống", "dữ liệu", "tài liệu", "tri thức", "bộ nhớ", "ký ức", "thực thể",
    "quan hệ", "chủ đề", "ngữ cảnh", "truy vấn", "tìm kiếm", "lưu trữ", "xử lý",
    "phân tích", "trích xuất", "chuẩn hoá", "chuẩn hóa", "chia đoạn", "lập chỉ mục",
    "chỉ mục", "tối ưu", "triển khai", "cấu hình", "cài đặt", "tích hợp", "kết nối",
    "đồng bộ", "sao lưu", "khôi phục", "bảo mật", "xác thực", "phân quyền", "giám sát",
    "kiểm thử", "đánh giá", "so sánh", "trực quan", "biểu đồ", "đồ thị", "giao diện",
    "màn hình", "chức năng", "tính năng", "ứng dụng", "phần mềm", "phần cứng", "hạ tầng",
    "máy chủ", "máy trạm", "mạng lưới", "điện toán", "đám mây", "dịch vụ", "giao thức",
    "mã nguồn", "lập trình", "phát triển", "vận hành", "bảo trì", "nâng cấp", "cập nhật",
    "phiên bản", "nhật ký", "cảnh báo", "lỗi mạng", "ngoại lệ", "thông báo", "hàng đợi",
    "tiến trình", "luồng chạy", "tác vụ", "công việc", "bàn giao", "hộp thư", "tin nhắn",
    "mô hình", "thuật toán", "ma trận", "véc tơ", "nhúng số", "phân cụm", "phân loại",
    "ngôn ngữ", "văn bản", "đoạn văn", "tiêu đề", "mục lục", "chứng cứ", "trích dẫn",
    "từ khoá", "từ khóa", "nhãn dán", "bộ lọc", "sắp xếp", "phân trang", "tốc độ",
    "độ trễ", "hiệu năng", "hiệu suất", "băng thông", "dung lượng", "tài nguyên",
    # 2 syllable compounds — business, operations & manufacturing
    "quy trình", "nguyên liệu", "vật liệu", "sản phẩm", "sản xuất", "nhà máy", "xưởng sản",
    "dây chuyền", "thiết bị", "máy móc", "vận chuyển", "kho hàng", "tồn kho", "nhập kho",
    "xuất kho", "kiểm kê", "kiểm soát", "chất lượng", "tiêu chuẩn", "quy chuẩn", "quy định",
    "chính sách", "kế hoạch", "chiến lược", "mục tiêu", "tiến độ", "dự án", "nhiệm vụ",
    "yêu cầu", "đề xuất", "giải pháp", "quyết định", "phương án", "rủi ro", "sự cố",
    "khắc phục", "báo cáo", "thống kê", "tổng hợp", "chi tiết", "tổng quan", "khảo sát",
    "nghiên cứu", "thử nghiệm", "thực tế", "kết quả", "hiệu quả", "chi phí", "ngân sách",
    "doanh thu", "lợi nhuận", "hợp đồng", "khách hàng", "đối tác", "nhà cung", "nhân viên",
    "nhân sự", "phòng ban", "đội ngũ", "trưởng nhóm", "quản lý", "giám đốc", "kỹ sư",
    "chuyên gia", "công ty", "doanh nghiệp", "tổ chức", "tập đoàn", "chi nhánh", "văn phòng",
    "thời gian", "lịch trình", "thời hạn", "hôm nay", "hôm qua", "ngày mai", "tuần này",
    "tuần trước", "tháng này", "tháng trước", "hiện tại", "tương lai", "lịch sử",
    "người dùng", "tài khoản", "mật khẩu", "quyền hạn", "vai trò", "phiên làm", "hồ sơ",
    "cà phê", "sữa đá", "trà sữa", "đồ uống", "ẩm thực", "du lịch", "giáo dục", "y tế",
    "tài chính", "ngân hàng", "thương mại", "bán hàng", "tiếp thị", "hỗ trợ", "tư vấn",
    "đào tạo", "hướng dẫn", "tài nguyên", "thông tin", "nội dung", "hình ảnh", "âm thanh",
    "video", "biểu mẫu", "danh sách", "bảng tính", "trạng thái", "kết nối", "phản hồi",
    "bình luận", "đóng góp", "cải tiến", "thay đổi", "chỉnh sửa", "xoá bỏ", "xóa bỏ",
    "tạo mới", "khởi tạo", "kết thúc", "hoàn thành", "hoàn tất", "thất bại", "thành công",
    "sẵn sàng", "đang chạy", "chờ xử", "tạm dừng", "huỷ bỏ", "hủy bỏ",
    "diễn biến", "diễn giải", "kiểu dữ", "bảo đảm", "duy trì", "gửi đi", "cân bằng",
    "cân nhắc", "định nghĩa", "định dạng", "định vị", "xác định", "quyết định", "ổn định",
    "liên kết", "liên quan", "tương tác", "tương thích", "phụ thuộc", "độc lập", "phân tán",
    "tập trung", "tự động", "thủ công", "trực tiếp", "gián tiếp", "cục bộ", "toàn cục",
    "đa ngôn", "song ngữ", "tiếng việt", "tiếng anh",
})

# Common Vietnamese stopwords (single syllables & function compounds)
VI_STOPWORDS: frozenset[str] = frozenset({
    "và", "của", "là", "các", "có", "được", "cho", "trong", "một", "những", "này",
    "khi", "với", "từ", "đến", "tại", "về", "như", "không", "đã", "sẽ", "đang", "theo",
    "qua", "sau", "trước", "để", "nếu", "thì", "hay", "hoặc", "bởi", "vì", "nên",
    "trên", "dưới", "giữa", "cùng", "cũng", "vẫn", "tôi", "mình", "bạn", "anh", "chị",
    "em", "chúng", "họ", "nó", "rất", "lắm", "quá", "nữa", "đều", "chỉ", "mới", "nào",
    "gì", "ai", "đâu", "sao", "thế", "vậy", "nhé", "nhỉ", "hãy", "đừng", "phải", "cần",
    "muốn", "lại", "rồi", "chưa", "từng", "mỗi", "mọi", "nhiều", "ít", "hơn", "nhất",
    "bằng", "đó", "đây", "kia", "ấy", "nhưng", "tuy", "ngoài", "vào", "lên", "xuống",
    "ra", "đi", "ở", "tất", "cả", "mà", "do", "tức", "gồm", "gọi", "bị", "sẵn", "được",
    "việc", "cách", "loại", "kiểu", "phần", "mục", "bước", "lần", "bản", "cấp", "độ",
    "số", "tên", "mã", "giá", "bên", "nơi", "lúc", "chỗ", "ngay", "luôn", "thường",
    "hay", "vừa", "xong", "hết", "đủ", "toàn", "riêng", "chung", "chính", "phụ", "khác",
    "giống", "nhau", "tự", "nên", "còn", "thêm", "bớt", "dùng", "làm", "xem", "biết",
    "thấy", "nói", "hỏi", "trả", "lời", "gửi", "nhận", "lấy", "đưa", "mở", "đóng",
    "bật", "tắt", "chọn", "nhập", "xuất", "tạo", "xoá", "xóa", "sửa", "đổi", "chạy",
    "dừng", "lưu", "tải", "nạp", "đọc", "viết", "gõ", "bấm", "nhấn", "kéo", "thả",
    # Unaccented folded versions
    "va", "cua", "la", "cac", "co", "duoc", "cho", "trong", "mot", "nhung", "nay",
    "khi", "voi", "tu", "den", "tai", "ve", "nhu", "khong", "da", "se", "dang", "theo",
    "qua", "sau", "truoc", "de", "neu", "thi", "hoac", "boi", "vi", "nen", "toi", "minh",
    "ban", "chi", "chung", "ho", "rat", "lam", "nua", "deu", "moi", "nao", "gi", "dau",
    "the", "vay", "nhe", "nhi", "dung", "phai", "can", "muon", "lai", "roi", "chua",
    "tung", "nhieu", "hon", "nhat", "bang", "do", "day", "ay", "nhung", "ngoai", "vao",
    "len", "xuong", "di", "ma", "bi", "san", "viec", "cach", "loai", "kieu", "phan",
    "muc", "buoc", "lan", "cap", "ten", "gia", "ben", "noi", "luc", "cho", "ngay",
    "luon", "thuong", "vua", "xong", "het", "du", "toan", "rieng", "khac", "nhau",
    "con", "them", "bot", "xem", "biet", "thay", "noi", "hoi", "gui", "nhan", "lay",
    "dua", "mo", "dong", "bat", "tat", "chon", "nhap", "xuat", "tao", "xoa", "sua",
    "doi", "chay", "luu", "tai", "nap", "doc", "viet", "go", "bam", "nhan", "keo", "tha",
})

# Common capitalized Vietnamese heading / table-cell / clause-start single syllables that are NOT entities.
# Directly eliminates review #14/#15/#16 noise like "Diễn", "Kiểu", "Ca", "Bảo", "Duy", "Gửi", "Sẵn", "Bị", "Nạp", "Phiếu", "Cân", "Đinh".
VI_COMMON_SINGLE_SYLLABLES: frozenset[str] = frozenset({
    "diễn", "kiểu", "ca", "bảo", "duy", "gửi", "sẵn", "bị", "nạp", "phiếu", "cân", "đinh",
    "định", "tạo", "xoá", "xóa", "sửa", "đọc", "ghi", "lưu", "tải", "mở", "đóng", "bật",
    "tắt", "chọn", "nhập", "xuất", "chạy", "dừng", "xem", "tìm", "lọc", "thêm", "bớt",
    "đổi", "gộp", "tách", "chia", "trộn", "đánh", "kiểm", "tra", "xử", "lý", "phân",
    "tích", "trích", "chuẩn", "hoá", "hóa", "lập", "chỉ", "mục", "phần", "bước", "trang",
    "đoạn", "dòng", "cột", "hàng", "bảng", "biểu", "đồ", "thị", "hình", "ảnh", "tệp",
    "thư", "tài", "liệu", "dữ", "tri", "thức", "ký", "ức", "bộ", "nhớ", "thực", "thể",
    "quan", "hệ", "thống", "chủ", "đề", "ngữ", "cảnh", "truy", "vấn", "kết", "quả",
    "trạng", "thái", "nhãn", "cụm", "nhóm", "mốc", "thời", "gian", "ngày", "tháng", "năm",
    "tuần", "giờ", "phút", "giây", "tổng", "số", "lượng", "giá", "trị", "tỷ", "lệ",
    "điểm", "hạng", "mức", "cấp", "độ", "loại", "dạng", "mẫu", "khối", "vùng", "khu",
    "vực", "phạm", "vi", "đích", "nguồn", "gốc", "chính", "phụ", "chung", "riêng", "mới",
    "cũ", "tốt", "xấu", "cao", "thấp", "lớn", "nhỏ", "nhanh", "chậm", "đúng", "sai",
    "đủ", "thiếu", "trống", "đầy", "lỗi", "cảnh", "báo", "chú", "ý", "ghi", "chú",
    "hướng", "dẫn", "quy", "trình", "vật", "sản", "phẩm", "dịch", "vụ", "khách",
    "nhân", "viên", "phòng", "ban", "công", "ty", "dự", "án", "nhiệm", "yêu", "cầu",
    "đề", "xuất", "giải", "pháp", "quyết", "phương", "rủi", "ro", "sự", "cố", "báo", "cáo",
    "kế", "hoạch", "chiến", "lược", "tiến", "hiệu", "năng", "suất", "tốc", "trễ", "chi",
    "phí", "ngân", "sách", "hợp", "đồng", "tiêu", "chuẩn", "chất", "toàn", "mật",
    "quyền", "vai", "trò", "phiên", "bản", "tài", "khoản", "người", "dùng", "quản", "trị",
    "nhà", "phát", "triển", "môi", "trường", "máy", "mạng", "cài", "đặt", "cấu", "hình",
    "tích", "kết", "nối", "đồng", "sao", "khôi", "phục", "giám", "sát", "thử", "nghiệm",
    "so", "sánh", "trực", "giao", "diện", "màn", "chức", "tính", "ứng", "dụng", "mềm",
    "cứng", "hạ", "tầng", "mã", "lập", "vận", "hành", "trì", "nâng", "cập", "nhật",
    "thông", "tin", "nội", "dung", "danh", "sách", "phản", "hồi", "bình", "luận", "đóng",
    "góp", "cải", "thay", "chỉnh", "khởi", "hoàn", "tất", "thành", "thất", "bại", "tạm",
    "huỷ", "hủy", "liên", "tương", "tác", "thích", "thuộc", "độc", "tán", "trung", "động",
    "thủ", "tiếp", "gián", "cục", "toàn", "đa", "song", "việt", "anh", "cà", "phê", "sữa",
    "đá", "trà", "nước", "điện", "nhiệt", "áp", "lực", "khí", "hơi", "dầu", "xăng", "than",
    "sắt", "thép", "gỗ", "giấy", "nhựa", "vải", "kính", "gạch", "đất", "cát", "đường",
    "muối", "bột", "hạt", "trộn", "sấy", "ép", "cắt", "hàn", "sơn", "rửa", "làm", "nguội",
    "đóng", "gói", "dán", "tem", "xuất", "nhập", "kho", "xe", "tàu", "cảng", "trạm",
    "kênh", "cổng", "luồng", "tác", "nhận", "giao", "trả", "hỏi", "đáp", "nói", "viết",
    "nghe", "nhìn", "biết", "hiểu", "nghĩ", "nhớ", "quên", "muốn", "cần", "phải", "nên",
    "có", "không", "chưa", "đã", "đang", "sẽ", "vẫn", "còn", "rồi", "xong", "hết", "lại",
    "nữa", "thêm", "ra", "vào", "lên", "xuống", "qua", "về", "đi", "đến", "tới", "tại",
    "ở", "trong", "ngoài", "trên", "dưới", "giữa", "bên", "cạnh", "gần", "xa", "trước", "sau",
    "đầu", "cuối", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "trăm", "nghìn", "triệu", "tỷ", "thứ", "nhất", "nhì", "nhiều", "ít", "vài", "mọi", "mỗi",
    "tất", "cả", "các", "những", "này", "đó", "kia", "đây", "ấy", "nào", "gì", "ai", "đâu",
    "sao", "nếu", "thì", "vì", "bởi", "do", "nhưng", "tuy", "mặc", "dù", "hoặc", "hay", "và",
    "cùng", "với", "của", "cho", "bằng", "theo", "như", "để", "mà", "rằng", "là", "được",
})


def is_capitalized_vi_compound_head(words: list[str], idx: int) -> bool:
    """Return True if words[idx] is merely the sentence-cased first syllable of a Vietnamese compound word
    (e.g., 'Diễn giải', 'Bảo trì', 'Nạp nguyên liệu', 'Phiếu kiểm tra', 'Nguyên liệu').
    """
    if idx < 0 or idx >= len(words):
        return False
    cur = words[idx].strip(",:;()[]{}\"'`").lower()
    if not cur:
        return False
    for span in (4, 3, 2):
        if idx + span <= len(words):
            tail = [words[k].strip(",:;()[]{}\"'`") for k in range(idx + 1, idx + span)]
            # Only treat as sentence-cased compound head when following syllables are lowercase
            if all(t and t[0].islower() for t in tail):
                cand = " ".join([cur, *(t.lower() for t in tail)])
                if cand in VI_COMPOUND_WORDS:
                    return True
    return False

EN_TOPIC_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "may", "me", "more", "most", "my", "no", "not", "of", "on", "or", "our", "out",
    "over", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "up", "us", "very", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "whose", "why", "will", "with", "would", "you", "your", "about",
    "also", "only", "still", "well", "both", "each", "use", "using", "used", "default",
    "file", "files", "data", "item", "items", "value", "values", "type", "name", "list",
    "run", "runs", "set", "get", "add", "new", "one", "two", "three", "see", "note",
})

_WORD_RE = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", flags=re.UNICODE)


def _fold_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).replace("đ", "d").replace("Đ", "D")


@lru_cache(maxsize=1)
def _load_optional_vi_backend():
    """Optionally use `pyvi` or `underthesea` if installed in the host environment."""
    try:
        from pyvi import ViTokenizer  # type: ignore[import-not-found]
        return ("pyvi", ViTokenizer.tokenize)
    except Exception:
        pass
    try:
        from underthesea import word_tokenize  # type: ignore[import-not-found]
        return ("underthesea", lambda s: word_tokenize(s, format="text"))
    except Exception:
        pass
    return (None, None)


def has_vietnamese_diacritics(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def segment_vietnamese(text: str) -> list[str]:
    """Segment text into Vietnamese compound words + English tokens (lowercased).

    Uses pyvi/underthesea when available, otherwise deterministic maximal-forward-matching
    against VI_COMPOUND_WORDS (supporting up to 4-syllable compounds).
    """
    if not text or not text.strip():
        return []

    backend_name, backend_fn = _load_optional_vi_backend()
    if backend_fn is not None and has_vietnamese_diacritics(text):
        try:
            segmented = str(backend_fn(text))
            units: list[str] = []
            for raw in segmented.split():
                clean = raw.strip(".,:;!?()[]{}\"'`/\\|<>~*").replace("_", " ").lower()
                if clean and len(clean) >= 2:
                    units.append(clean)
            if units:
                return units
        except Exception:
            pass

    raw_tokens = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    if not raw_tokens:
        return []

    result: list[str] = []
    i = 0
    n = len(raw_tokens)
    while i < n:
        matched = None
        for span in (4, 3, 2):
            if i + span <= n:
                cand = " ".join(raw_tokens[i : i + span])
                if cand in VI_COMPOUND_WORDS:
                    matched = cand
                    i += span
                    break
        if matched is not None:
            result.append(matched)
        else:
            result.append(raw_tokens[i])
            i += 1
    return result


def extract_topic_phrases(text: str, *, max_phrases: int = 30) -> list[str]:
    """Extract meaningful Vietnamese/English phrases for c-TF-IDF topic modeling.

    Produces clean multi-word phrases (segmented Vietnamese compounds + adjacent content bigrams)
    and meaningful technical tokens, strictly filtering stopwords and bare single syllables.
    """
    units = segment_vietnamese(text)
    if not units:
        return []

    def _is_stop(u: str) -> bool:
        if len(u) < 2 or u.isdigit():
            return True
        if u in VI_STOPWORDS or u in EN_TOPIC_STOPWORDS:
            return True
        folded = _fold_ascii(u)
        if " " not in u and (folded in VI_STOPWORDS or folded in EN_TOPIC_STOPWORDS):
            return True
        return False

    phrases: list[str] = []
    content_positions: list[tuple[int, str]] = []

    for idx, unit in enumerate(units):
        if _is_stop(unit):
            continue
        content_positions.append((idx, unit))
        # Multi-word Vietnamese compounds from the lexicon are already strong topic phrases
        if " " in unit:
            phrases.append(unit)
        # Technical/English words with >= 4 chars (or known tech like sql/api/rag/llm/mcp) are strong unigrams
        elif not has_vietnamese_diacritics(unit) and (len(unit) >= 4 or unit in {"api", "sql", "rag", "llm", "mcp", "rrf", "gpu", "cpu", "tls", "jwt", "s3"}):
            phrases.append(unit)

    # Combine adjacent content units into 2-unit phrases (e.g. "quy trình" + "nạp nguyên liệu", "docker" + "postgresql")
    for k in range(len(content_positions) - 1):
        pos_a, ua = content_positions[k]
        pos_b, ub = content_positions[k + 1]
        if pos_b != pos_a + 1 or ua == ub:
            continue
        combined = f"{ua} {ub}"
        word_count = len(combined.split())
        if 2 <= word_count <= 4:
            phrases.append(combined)

    # Keep order & frequency (for TF-IDF counting) bounded
    return phrases[: max_phrases * 4]


def is_single_syllable_vi_noise(name: str) -> bool:
    """Return True if `name` is a single Vietnamese syllable that is a common word/noise rather than an entity."""
    cleaned = name.strip().lower()
    if not cleaned or " " in cleaned:
        return False
    if cleaned in VI_COMMON_SINGLE_SYLLABLES or cleaned in VI_STOPWORDS:
        return True
    # Any single short accented syllable (<= 4 chars) is almost always a broken Vietnamese word unless explicitly known
    if has_vietnamese_diacritics(cleaned) and len(cleaned) <= 4:
        return True
    return False


def extract_evidence_sentence(text: str, entity_name: str, *, max_len: int = 220) -> str:
    """Extract the best real sentence from `text` containing `entity_name` for Entity Table evidence."""
    if not text or not entity_name:
        return ""
    needle = entity_name.strip().lower()
    needle_folded = _fold_ascii(needle)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    for sent in sentences:
        low = sent.lower()
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", low) or (
            len(needle_folded) >= 3 and re.search(rf"(?<!\w){re.escape(needle_folded)}(?!\w)", _fold_ascii(low))
        ):
            clean = " ".join(sent.split())
            if len(clean) > max_len:
                return clean[: max_len - 1].rstrip() + "…"
            return clean
    return ""
