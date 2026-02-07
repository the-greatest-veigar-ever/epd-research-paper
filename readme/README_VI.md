# Kiến trúc Phòng thủ Đa tác nhân Lai (Hybrid Multi-Agent Defense Architecture - EPD)

## Tổng quan
Dự án này mô phỏng một **Kiến trúc Phòng thủ Đa tác nhân Lai** được thiết kế để giảm thiểu rủi ro **Quyền hạn Quá mức (Excessive Agency)** và **Bề mặt Tấn công Nhận thức Tĩnh (Static Cognitive Attack Surfaces)** trong các tác nhân bảo mật đám mây. Hệ thống chuyển đổi từ các bot bảo mật thường trực sang mô hình **Phòng thủ Đa hình Phù du (Ephemeral Polymorphic Defense - EPD)**.

Hệ thống điều phối ba nhóm tác nhân riêng biệt để phát hiện, phân tích và khắc phục các mối đe dọa trên đám mây, sử dụng **Phòng thủ Mục tiêu Di động (Moving Target Defense - MTD)** ở lớp nhận thức.

## Kết quả Nghiên cứu Chính (Nghiên cứu Loại bỏ Q1)
Chúng tôi đã thực hiện một mô phỏng nghiêm ngặt (N=2000) so sánh kiến trúc này với các tiêu chuẩn truyền thống.

| Chỉ số | Tiêu chuẩn Thường trực | **EPD Đầy đủ (Của chúng tôi)** | Cải thiện |
| :--- | :--- | :--- | :--- |
| **Tỷ lệ Thành công (Phòng thủ)** | 60.4% | **67.6%** | **+7.2%** |
| **Ý nghĩa Thống kê** | - | **p = 0.0177** | Có ý nghĩa (p < 0.05) |

> **Kết luận**: Sự kết hợp giữa việc Khởi tạo Theo nhu cầu (Just-in-Time), Xoay vòng Mô hình (Model Rotation) và Đột biến Lời nhắc (Prompt Mutation) cải thiện đáng kể khả năng chống lại các cuộc tấn công "Bẻ khóa" (Jailbreak) và "Tích lũy Ngữ cảnh" (Context Accumulation).

## Các Lớp Kiến trúc

### Nhóm 1: Người Quan sát (Thường trực)
*   **Vai trò**: Giám sát liên tục.
*   **Logic**: Quét các nhật ký (GuardDuty/CloudTrail) để tìm kiếm sự bất thường.
*   **Cài đặt**: `monitor.py`

### Nhóm 2: Bộ não (Lai)
*   **Vai trò**: Tình báo và Lập kế hoạch.
*   **Logic**: Sử dụng cơ chế đồng thuận (SentinelNet) để xác thực các mối đe dọa và kiểm tra chính sách an toàn (VeriGuard) để phê duyệt kế hoạch.
*   **Cài đặt**: `intelligence.py`

### Nhóm 3: Tác nhân Bóng ma (Phù du)
*   **Vai trò**: Thực thi.
*   **Logic**:
    *   **Theo nhu cầu (Just-in-Time)**: Chỉ được tạo ra khi cần thiết.
    *   **Đa hình (Polymorphic)**: Xoay vòng các LLM (GPT-4o, Claude, v.v.) và làm biến đổi các lời nhắc hệ thống (system prompts) để ngăn chặn việc bẻ khóa.
    *   **Tự sát (Suicide)**: Tự hủy ngay lập tức sau khi hoàn thành nhiệm vụ.
*   **Cài đặt**: `epd_core.py`

## Cấu trúc Thư mục

```text
├── README.md               # Tài liệu dự án (Tiếng Anh)
├── README_VI.md            # Tài liệu dự án (Tiếng Việt)
├── src/                    # Mã nguồn
│   ├── main.py             # Bản demo tương tác
│   ├── research_sim.py     # Mô phỏng Nghiên cứu Loại bỏ Q1
│   └── ...
└── Simulation Test/        # Dữ liệu & Tạo phẩm Nghiên cứu
    ├── 01_Baseline_vs_EPD  # Các thử nghiệm ban đầu
    └── 02_Q1_Ablation_Study# Bộ dữ liệu nghiên cứu đầy đủ
        ├── EPD_Q1_Research_Data.xlsx
        └── architecture.mermaid
```

## Cách chạy

### 1. Bản demo tương tác
Mô phỏng vòng đời của một cuộc tấn công đơn lẻ (Phát hiện -> Đồng thuận -> Khắc phục):
```bash
python3 src/main.py
```

### 2. Mô phỏng Nghiên cứu (Nghiên cứu Loại bỏ)
Tái tạo đánh giá thống kê N=2000:
```bash
python3 src/research_sim.py
```

## Công việc Tương lai
*   Tích hợp với các API trực tiếp của AWS/Azure.
*   Mở rộng thư viện Đột biến Lời nhắc (Prompt Mutation).
*   Thử nghiệm đối kháng với các mô hình tấn công mới hơn (ví dụ: mô phỏng GPT-5).

---

## Bộ Dữ liệu Nghiên cứu

### Nhóm 1: The Watchers (Hệ thống Giám sát)
**Nhiệm vụ:** Phát hiện xâm nhập mạng (Intrusion Detection)
*   **Bộ dữ liệu:** CSE-CIC-IDS2018 (AWS & Đại học New Brunswick)
*   **Quy mô:** ~6.4 GB CSV đã xử lý
*   **Kết quả:** Watcher Agent đạt tỷ lệ phòng thủ **67.7%**

### Nhóm 2: The Brain (Bộ não Trung tâm)
**Nhiệm vụ:** Tư duy logic, kiểm tra chéo và ra quyết định
*   **MMLU:** Bảo mật máy tính, Logic hình thức, Học máy
*   **HaluEval:** 35,000+ mẫu kiểm tra ảo giác

### Nhóm 3: Ghost Agents (Agent Bóng ma)
**Nhiệm vụ:** Thực thi đối kháng, tự vệ trước Jailbreak/Prompt Injection
*   **Bộ dữ liệu:** Agent Security Bench (ASB) - ICLR 2025
*   **Kết quả:** Ghost Agent đạt tỷ lệ kháng cự **85.4%** (gấp đôi Agent tĩnh 39%)

