# Kịch bản Tổng hợp Dữ liệu Nghiên cứu (Research Datasets) cho Kiến trúc EPD

Dưới đây là danh sách các bộ dữ liệu chuẩn (Standard Benchmarks) được sử dụng để kiểm chứng hiệu quả của 3 nhóm Agent trong kiến trúc EPD, phục vụ cho bài báo Q1.

---

## 1. Nhóm 1: The Watchers (Hệ thống Giám sát)
**Nhiệm vụ:** Phát hiện xâm nhập mạng (Intrusion Detection) dựa trên lưu lượng (Network Traffic).

*   **Tên bộ dữ liệu:** **CSE-CIC-IDS2018** (Do AWS & Đại học New Brunswick phát hành).
*   **Quy mô:** Sử dụng tập dữ liệu đã xử lý (~6.4 GB file CSV).
*   **Mục đích sử dụng:**
    *   Giả lập môi trường mạng thực tế với hàng triệu gói tin.
    *   Chứa các mẫu tấn công hiện đại: DDoS, Botnet, Brute Force, Infiltration.
    *   **Kết quả:** Watcher Agent đạt tỷ lệ phòng thủ **67.7%** (vượt trội so với Baseline 60.2%).

---

## 2. Nhóm 2: The Brain (Bộ não Trung tâm)
**Nhiệm vụ:** Tư duy logic, kiểm tra chéo (Consensus) và ra quyết định xử lý.

Chúng tôi sử dụng kết hợp 2 bộ dữ liệu để đánh giá "Trí tuệ" và "Độ tin cậy":

1.  **MMLU (Massive Multitask Language Understanding):**
    *   **Thành phần:** Gồm các tập con về *Bảo mật máy tính (Computer Security)*, *Logic hình thức* và *Học máy*.
    *   **Mục đích:** Chứng minh Agent "Brain" có kiến thức chuyên môn sâu rộng để phân tích mối đe dọa.
2.  **HaluEval (Hallucination Evaluation):**
    *   **Quy mô:** Hơn 35,000 mẫu.
    *   **Mục đích:** Kiểm tra khả năng phát hiện ảo giác (thông tin sai lệch). Đây là cốt lõi của cơ chế **Consensus** (Đồng thuận) - giúp hệ thống không bị đánh lừa bởi các cảnh báo giả.

---

## 3. Nhóm 3: Ghost Agents (Agent Bóng ma)
**Nhiệm vụ:** Thực thi đối kháng, tự vệ trước các cuộc tấn công vào Agent (Jailbreak, Prompt Injection).

*   **Tên bộ dữ liệu:** **Agent Security Bench (ASB)**.
*   **Nguồn:** Công trình mới nhất tại hội nghị ICLR 2025 (Top-tier AI Conference).
*   **Mục đích sử dụng:**
    *   Đánh giá khả năng chống lại các cuộc tấn công tiêm nhiễm (Injection Attacks) vào bộ nhớ và quy hoạch (Planning).
    *   **Kết quả:** Nhờ cơ chế **Đa hình (Polymorphism)** (tự động thay đổi Prompt và Model), Ghost Agent đạt tỷ lệ kháng cự **85.4%**, cao hơn gấp đôi so với Agent tĩnh thông thường (39%).

---

**Tổng kết:**
Việc kết hợp **CSE-CIC-IDS2018** (Dữ liệu mạng kinh điển) với **MMLU & ASB** (Các chuẩn đánh giá LLM hiện đại nhất 2024-2025) tạo nên độ tin cậy vững chắc và tính mới (Novelty) cho bài báo nghiên cứu.
