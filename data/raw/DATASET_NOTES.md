# Ghi chú dataset Zhou et al. (LLM4DR)

Nguồn: https://github.com/Eric0052/LLM4DR

## Cấu trúc file đã tải

- `Collected_Data.xlsx` — 100 bài toán kiến trúc gốc (SO/GitHub/Discussion), có cột
  `Design Rationale (Human Experts)` (DR tham chiếu do chuyên gia viết).
- `Code/Architecture_Context.xlsx` — 100 dòng, cột `Reference Evaluation` = loại quyết định
  kiến trúc (Implementation/Existence/... Decision).
- `Results/zero-shot.xlsx`, `Results/CoT.xlsx`, `Results/AI-Agent.xlsx` — mỗi file có 5 sheet
  (theo model: gpt-3.5-turbo, gpt-4-0613, gemini-1.0-pro, llama3-8B, mistral-7B), mỗi sheet
  100 dòng. Cột quan trọng:
  - `Design Rationale (...)` — DR do LLM sinh ra (đoạn văn, thường chứa NHIỀU luận điểm/câu)
  - `Insightful` / `Helpful` / `Uncertain` / `Misleading` — **SỐ LƯỢNG luận điểm** trong DR đó
    thuộc mỗi nhãn (không phải nhãn cho từng câu riêng lẻ, và không có text luận điểm được tách sẵn)
  - `Consistent Arguments (LLM-generated)` vs `Number of Arguments (Human Experts)` → dùng để
    tính Precision/Recall/F1 gốc của Zhou et al.
- `Pilot_Results/Pilot_Results.xlsx` — dữ liệu pilot (agreement giữa 2 người gán nhãn), chỉ
  16-22 dòng, KHÔNG phải toàn bộ dataset.

## ⚠️ Phát hiện quan trọng — cần quyết định trước khi code Verifier

Nhãn IHUM gốc của Zhou et al. là **đếm số luận điểm** trong mỗi đoạn DR (do người đọc tay và
đếm), KHÔNG phải nhãn categorical gắn cho từng câu văn cụ thể có text đi kèm. Điều này khác với
mô tả ban đầu trong brief ("mỗi câu DR ứng với 1 nhãn IHUM").

Nghĩa là: dataset công khai không có sẵn cặp (câu luận điểm, nhãn) — chỉ có (đoạn DR đầy đủ,
tổng số câu Insightful/Helpful/Uncertain/Misleading trong đoạn đó).

Ảnh hưởng tới thiết kế Verifier Agent — 2 hướng khả thi:

1. **Per-DR aggregate (khớp thẳng với dataset có sẵn):** Verifier đọc toàn bộ đoạn DR, tự
   tách thành các luận điểm, tự gán nhãn từng luận điểm, rồi cộng lại thành số lượng
   I/H/U/M dự đoán cho cả đoạn → so với số lượng ground-truth per DR. Hoặc đơn giản hơn:
   Verifier trả lời nhị phân "đoạn DR này CÓ chứa luận điểm Misleading hay không?" so với
   ground-truth `Misleading > 0`. Cách này an toàn, không cần thêm bước gán nhãn thủ công.

2. **Per-argument granularity (đúng như brief mô tả ban đầu):** Cần tự tách từng đoạn DR
   thành các câu/luận điểm riêng, rồi thủ công (hoặc nhờ 1 LLM khác + review tay) gán lại
   nhãn IHUM cho từng câu để có ground-truth mới ở mức câu. Tốn thêm thời gian gán nhãn,
   rủi ro tiến độ với deadline 20/7.

Đã hỏi người dùng chọn hướng nào trước khi viết `verifier_agent.py`.
