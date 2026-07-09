# Tổng quan đồ án tốt nghiệp: Clinical Auditor AI Agent

## 1. Tên đề tài

**Hệ thống AI Agent hỗ trợ kiểm tra đơn thuốc và cảnh báo tương tác thuốc đa nguồn**

Tên ngắn trong quá trình phát triển:

**Clinical Auditor AI Agent**

## 2. Bản chất của hệ thống

Clinical Auditor AI Agent là một hệ thống **AI Agent kết hợp Retrieval-Augmented Generation (RAG) và Doctor Memory**, được xây dựng nhằm hỗ trợ bác sĩ, dược sĩ và nhân viên y tế kiểm tra đơn thuốc dựa trên bằng chứng y khoa.

Hệ thống không phải là chatbot hỏi đáp y tế thông thường và cũng không phải là hệ thống RAG thuần. Điểm chính của hệ thống là thực hiện một quy trình kiểm tra đơn thuốc nhiều bước, có kiểm soát:

```text
Đơn thuốc / câu hỏi chuyên môn
→ nhận diện ý định
→ trích xuất thuốc, hoạt chất, bệnh nền và ngữ cảnh liên quan
→ chuẩn hóa thuốc theo hướng generic-first
→ ánh xạ hoạt chất sang kho bằng chứng y khoa
→ truy xuất evidence từ Qdrant
→ truy xuất ghi chú chuyên môn riêng của bác sĩ nếu có
→ phân tích nguy cơ bằng LLM dựa trên evidence
→ kiểm tra an toàn đầu ra
→ sinh báo cáo cảnh báo có nguồn
```

Trong đó:

- **RAG** giúp hệ thống truy xuất bằng chứng y khoa từ kho dữ liệu thuốc.
- **AI Agent** giúp điều phối nhiều bước xử lý: phân loại yêu cầu, đọc đơn, chuẩn hóa thuốc, truy xuất bằng chứng, phân tích nguy cơ, kiểm tra an toàn và sinh báo cáo.
- **Doctor Memory** cho phép bác sĩ lưu lại ghi chú chuyên môn riêng theo tài khoản, phục vụ các lần kiểm tra sau.

## 3. Phạm vi đồ án tốt nghiệp

Phiên bản final của đồ án tập trung vào MVP có thể triển khai và demo ổn định trong phạm vi sinh viên tốt nghiệp.

### 3.1. Phạm vi có trong phiên bản final

Hệ thống tập trung vào các chức năng chính sau:

1. Bác sĩ/dược sĩ đăng nhập hệ thống.
2. Nhập đơn thuốc dạng text theo format đã chuẩn hóa.
3. Hệ thống phân tích đơn thuốc, nhận diện thuốc và hoạt chất.
4. Chuẩn hóa thuốc theo hướng generic-first.
5. Truy xuất bằng chứng y khoa từ Qdrant.
6. Phân tích các nguy cơ liên quan đến đơn thuốc.
7. Sinh báo cáo cảnh báo có nguồn.
8. Hỏi đáp đơn lẻ về thuốc, tương tác thuốc, chống chỉ định, thận trọng.
9. Lưu và truy xuất ghi chú chuyên môn riêng của bác sĩ.
10. Chặn câu hỏi ngoài phạm vi y dược.

### 3.2. Phạm vi tạm thời chưa làm trong MVP

Trong phiên bản final của đồ án, tạm thời **chưa triển khai OCR/upload ảnh/PDF đơn thuốc**.

Lý do:

- OCR làm tăng độ phức tạp của hệ thống.
- Đọc ảnh/PDF cần xử lý nhiều lỗi thực tế như ảnh mờ, chữ viết tay, sai bố cục, thiếu thông tin.
- Mục tiêu chính của đồ án là chứng minh năng lực xây dựng AI Agent kiểm tra đơn thuốc dựa trên RAG và evidence y khoa.
- Với đồ án sinh viên, nhập text giúp kiểm soát đầu vào tốt hơn, dễ đánh giá pipeline AI Agent hơn.

OCR/upload PDF có thể được đưa vào phần **hướng phát triển** sau đồ án.

## 4. Đối tượng sử dụng

Hệ thống hướng đến:

- Bác sĩ
- Dược sĩ
- Nhân viên y tế cần hỗ trợ kiểm tra đơn thuốc

Hệ thống **không hướng đến bệnh nhân tự sử dụng để tự chẩn đoán, tự kê đơn hoặc tự thay đổi thuốc**.

Vai trò của hệ thống là hỗ trợ:

- Kiểm tra nguy cơ trong đơn thuốc.
- Phát hiện tương tác thuốc.
- Tra cứu bằng chứng y khoa.
- Gợi ý điểm cần bác sĩ/dược sĩ xem xét thêm.
- Lưu kinh nghiệm chuyên môn riêng của bác sĩ.

Hệ thống không thay thế bác sĩ/dược sĩ và không đưa ra quyết định điều trị cuối cùng.

## 5. Mục tiêu của đồ án

## 5.1. Mục tiêu tổng quát

Xây dựng một hệ thống AI Agent có khả năng hỗ trợ kiểm tra đơn thuốc và cảnh báo nguy cơ tương tác thuốc dựa trên dữ liệu y khoa đa nguồn, phù hợp với bối cảnh sử dụng tại Việt Nam.

## 5.2. Mục tiêu cụ thể

Hệ thống cần đạt được các mục tiêu sau:

1. Xây dựng kho dữ liệu y khoa đã được crawl, làm sạch, chunk, embedding và lưu vào Qdrant.
2. Xây dựng pipeline chuẩn hóa thuốc từ đơn thuốc sang hoạt chất.
3. Thiết kế cơ chế truy xuất evidence theo hoạt chất và section y khoa.
4. Tích hợp LLM để phân tích nguy cơ dựa trên evidence đã truy xuất.
5. Xây dựng workflow AI Agent có nhiều bước xử lý rõ ràng.
6. Xây dựng cơ chế Doctor Memory để lưu ghi chú riêng của bác sĩ.
7. Xây dựng Safety Layer để giới hạn phạm vi trả lời.
8. Xây dựng giao diện demo để nhập đơn thuốc, xem báo cáo và lưu note.

## 6. Định hướng xử lý thuốc: Generic-first

Một quyết định thiết kế quan trọng của hệ thống là xử lý thuốc theo hướng **generic-first**.

Điều này có nghĩa là hệ thống ưu tiên hoạt chất hơn tên thương mại.

Ví dụ đơn thuốc:

```text
Omeprazol (Kagascdine) 20mg
Metformin (Panfor SR) 750mg
Paracetamol (Hapacol Caplet) 500mg
```

Hệ thống sẽ ưu tiên:

```text
Omeprazol
Metformin
Paracetamol
```

để truy xuất bằng chứng y khoa, còn tên thương mại trong ngoặc chỉ dùng để kiểm tra chéo nếu cần.

Thứ tự xử lý:

```text
Ưu tiên 1: Hoạt chất được ghi trực tiếp trong đơn
Ưu tiên 2: Chuẩn hóa hoạt chất sang evidence slug
Ưu tiên 3: Dùng tên thương mại để kiểm tra chéo nếu cần
Ưu tiên 4: Nếu đơn chỉ ghi tên thương mại, dùng Long Châu để map sang hoạt chất
Ưu tiên 5: Nếu vẫn không xác định được, yêu cầu bác sĩ/dược sĩ xác nhận
```

Cách này giúp hệ thống không bị phụ thuộc quá nhiều vào việc tên thương mại có tồn tại trong một nguồn dữ liệu cụ thể hay không.

## 7. Các chức năng chính

## 7.1. Kiểm tra đơn thuốc dạng text

Bác sĩ/dược sĩ nhập đơn thuốc dạng text.

Ví dụ:

```text
ĐƠN NGOẠI TRÚ 1
Bệnh viện: Bệnh viện A
Khoa: Tiêu hóa
Đơn thuốc
I.THÔNG TIN BỆNH NHÂN
Họ và tên: Hoàng Thị P. 		
Tuổi: 28 		
Nam/Nữ: Nữ 
Cân nặng: 60kg
Địa chỉ: Nga Sơn, Thanh Hóa
II. THÔNG TIN LÂM SÀNG
Chẩn đoán: Viêm phế quản/loét dạ dày tá tràng
Dị ứng thuốc: Không
Bệnh nền: Không ghi nhận
Chức năng gan: Bình thường
Chức năng thận: Bình thường
Thai kỳ/ cho con bú: Cho con bú
Thuốc khác đang dùng: Không
III. CHỈ ĐỊNH DÙNG THUỐC
1.	Omeprazole (Losec) 20mg			x	15 viên 
Ngày uống 1 lần, mỗi lần 1 viên
2.	Sucralfate (Sucrate Gel) 1g/5mL		x	15 gói 
Ngày uống 3 lần, mỗi lần 1 gói
3.	Levofloxacine 500mg 			x	7 viên 
Ngày uống 1 viên

					Ngày , tháng, năm
					Bác sĩ khám bệnh
```

Hệ thống thực hiện:

```text
Nhận đơn thuốc
→ tách thông tin bệnh nhân và thông tin lâm sàng
→ tách từng dòng thuốc
→ nhận diện hoạt chất, tên thương mại, hàm lượng, cách dùng
→ chuẩn hóa hoạt chất
→ truy xuất evidence
→ phân tích nguy cơ
→ sinh báo cáo
```

## 7.2. Hỏi đáp đơn lẻ về thuốc

Người dùng có thể hỏi các câu như:

```text
Aspirin có tương tác với Warfarin không?
Metformin dùng cho bệnh nhân suy thận cần lưu ý gì?
Paracetamol có chống chỉ định gì?
Omeprazol có tương tác với Clopidogrel không?
```

Hệ thống sẽ:

```text
Nhận diện intent
→ trích xuất hoạt chất
→ chuẩn hóa hoạt chất
→ truy xuất evidence từ Qdrant
→ phân tích bằng Gemini dựa trên evidence
→ trả lời có nguồn
```

## 7.3. Doctor Memory

Doctor Memory là cơ chế lưu và truy xuất **ghi chú chuyên môn riêng của từng bác sĩ** sau khi bác sĩ đọc báo cáo kiểm tra đơn thuốc.

Ý tưởng chính không phải là để Doctor Memory thay thế bằng chứng y khoa, mà để hệ thống tận dụng kinh nghiệm cá nhân của bác sĩ trong các lần phân tích sau. Sau khi xem báo cáo, nếu bác sĩ thấy một cảnh báo, một hoạt chất, một nhóm bệnh nhân hoặc một tình huống lâm sàng cần lưu ý thêm, bác sĩ có thể ghi note lại.

Ví dụ note:

```text
Với bệnh nhân cao tuổi dùng aspirin và thuốc chống đông, cần chú ý nguy cơ xuất huyết và nên kiểm tra kỹ các yếu tố làm tăng nguy cơ chảy máu.
```

Mỗi note được gắn với:

```json
{
  "doctor_id": "doctor_001",
  "active_ingredients": ["aspirin", "warfarin"],
  "patient_context": ["người cao tuổi", "nguy cơ xuất huyết"],
  "note": "Với bệnh nhân cao tuổi dùng aspirin và thuốc chống đông, cần chú ý nguy cơ xuất huyết và nên kiểm tra kỹ các yếu tố làm tăng nguy cơ chảy máu.",
  "created_at": "2026-06-21"
}
```

Cách sử dụng trong lần phân tích sau:

```text
Đơn thuốc / câu hỏi mới
→ hệ thống nhận diện hoạt chất và ngữ cảnh bệnh nhân
→ truy xuất evidence y khoa từ Qdrant
→ tìm note riêng của bác sĩ theo doctor_id, hoạt chất và ngữ cảnh liên quan
→ nếu có note phù hợp: đưa note vào ngữ cảnh bổ sung cho LLM
→ nếu không có note phù hợp: chỉ phân tích dựa trên evidence y khoa như bình thường
```

Nguyên tắc:

- Doctor Memory là lớp thông tin bổ sung, không phải nguồn evidence y khoa chính.
- LLM phải ưu tiên bằng chứng y khoa từ kho evidence chung.
- Note của bác sĩ chỉ được dùng khi liên quan đến hoạt chất hoặc ngữ cảnh đang phân tích.
- Nếu hoạt chất/ngữ cảnh không có note phù hợp, hệ thống bỏ qua Doctor Memory và đánh giá như workflow RAG bình thường.
- Mỗi bác sĩ chỉ xem và truy xuất được note của chính mình.
- `doctor_id` lấy từ JWT token.
- Frontend không được tự truyền `doctor_id`.
- Doctor Memory phải được hiển thị tách biệt với nguồn evidence y khoa chung trong báo cáo.

## 7.4. Chặn câu hỏi ngoài phạm vi

Hệ thống chỉ hỗ trợ các tác vụ liên quan đến:

- Kiểm tra đơn thuốc.
- Tra cứu thuốc.
- Tương tác thuốc.
- Chống chỉ định.
- Thận trọng.
- Tác dụng không mong muốn.
- Thai kỳ/cho con bú.
- Suy gan/suy thận.
- Ghi chú chuyên môn của bác sĩ.

Nếu người dùng hỏi ngoài phạm vi, hệ thống sẽ từ chối.

Ví dụ:

```text
Xin lỗi, hệ thống này chỉ hỗ trợ các tác vụ liên quan đến kiểm tra đơn thuốc, tra cứu thông tin thuốc và cảnh báo tương tác thuốc. Câu hỏi hiện tại nằm ngoài phạm vi hỗ trợ của hệ thống.
```

## 8. Nguồn dữ liệu

Hệ thống sử dụng dữ liệu đa nguồn nhưng có phân vai rõ ràng.

## 8.1. Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam

Đây là nguồn evidence chính của hệ thống.

Vai trò:

- Cung cấp thông tin theo hoạt chất.
- Chỉ định.
- Chống chỉ định.
- Thận trọng.
- Liều dùng.
- Tác dụng không mong muốn.
- Tương tác thuốc.
- Thai kỳ/cho con bú.
- Suy gan/suy thận.
- Quá liều và xử trí.

Dữ liệu từ nguồn này được crawl, làm sạch, chunk theo section, embedding và lưu vào Qdrant.

## 8.2. Long Châu

Long Châu không phải nguồn evidence trung tâm.

Vai trò của Long Châu:

- Fallback khi đơn thuốc chỉ ghi tên thương mại.
- Ánh xạ tên thương mại sang hoạt chất.
- Kiểm tra chéo brand nếu đơn có ghi hoạt chất kèm tên thương mại.

Ví dụ:

```text
Hapacol 500mg → Paracetamol
Concor 5mg → Bisoprolol
Diamicron MR 60mg → Gliclazid
```

Nếu đơn đã ghi rõ hoạt chất, hệ thống không bắt buộc phải tìm được tên thương mại trong Long Châu.

## 8.3. Nguồn bổ sung

Sau khi hệ thống chạy ổn định tôi có thể thêm bộ data về dược chất của Long Châu làm nguồn bằng chứng nhưng dược thư của trung tâm thuốc vẫn uy tín hơn.

Tuy nhiên, với chức năng kiểm tra đơn thuốc, nguồn chính vẫn là Dược thư/Trung Tâm Thuốc.

## 9. Pipeline dữ liệu

Pipeline xây dựng dữ liệu gồm các bước:

```text
Crawl dữ liệu
→ lưu raw data
→ làm sạch dữ liệu
→ chuẩn hóa dữ liệu
→ chia chunk theo section
→ embedding bằng multilingual-e5-base
→ lưu vector vào Qdrant
→ xây evidence ingredient catalog
→ phục vụ truy xuất evidence
```

## 9.1. Raw data

Dữ liệu sau crawl được lưu lại ở dạng raw để có thể kiểm tra, debug và tái xử lý khi cần.

## 9.2. Làm sạch dữ liệu

Các bước làm sạch chính:

- Xóa HTML tag.
- Xóa menu, footer, quảng cáo.
- Xóa breadcrumb.
- Chuẩn hóa Unicode.
- Chuẩn hóa khoảng trắng và xuống dòng.
- Loại bỏ nội dung trùng lặp.
- Giữ lại URL nguồn.
- Giữ cấu trúc section y khoa.

## 9.3. Chunking theo section

Dữ liệu không được embedding nguyên bài dài, mà chia theo section.

Ví dụ với Paracetamol:

```text
paracetamol - chỉ định
paracetamol - chống chỉ định
paracetamol - thận trọng
paracetamol - tương tác thuốc
paracetamol - tác dụng không mong muốn
paracetamol - thai kỳ/cho con bú
paracetamol - quá liều và xử trí
```

Mỗi chunk có metadata:

```json
{
  "chunk_id": "trungtamthuoc:ingredient:paracetamol:tuong_tac_thuoc:001",
  "source": "trungtamthuoc",
  "source_name": "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam",
  "entity_type": "ingredient",
  "entity_name": "Paracetamol",
  "slug": "paracetamol",
  "section": "tuong_tac_thuoc",
  "language": "vi",
  "url": "...",
  "text": "..."
}
```

## 9.4. Evidence Ingredient Catalog

Sau khi có các chunk, hệ thống xây dựng catalog hoạt chất để phục vụ chuẩn hóa.

Ví dụ:

```json
{
  "catalog_id": "trungtamthuoc:ingredient:metformin",
  "entity_name": "Metformin",
  "slug": "metformin",
  "normalized_name": "metformin",
  "aliases": ["metformin"],
  "sections": ["chi_dinh", "chong_chi_dinh", "than_trong", "tuong_tac_thuoc"],
  "chunk_count": 64,
  "url": "..."
}
```

Catalog giúp hệ thống resolve hoạt chất trong đơn sang đúng `evidence_slug`.

## 10. Các module backend chính

## 10.1. Auth Service

Phụ trách:

- Đăng nhập bác sĩ.
- Sinh JWT token.
- Lấy `doctor_id` từ token.
- Bảo vệ API cần đăng nhập.

## 10.2. Intent Router

Phân loại yêu cầu của người dùng.

Các intent chính:

```text
PRESCRIPTION_CHECK
DRUG_INTERACTION_QUERY
SINGLE_DRUG_QUERY
DRUG_DISEASE_QUERY
DRUG_FOOD_QUERY
DOCTOR_NOTE_CREATE
DOCTOR_NOTE_QUERY
OUT_OF_SCOPE
```

## 10.3. Prescription Parser

Trong MVP hiện tại, parser nhận đầu vào text đã có cấu trúc.

Nhiệm vụ:

- Tách thông tin bệnh nhân.
- Tách thông tin lâm sàng.
- Tách danh sách thuốc.
- Chuẩn hóa thành JSON nội bộ.

## 10.4. MedicationLineParser

Tách từng dòng thuốc thành cấu trúc rõ ràng.

Ví dụ input:

```text
Spiramycin + Metronidazol (Spirastad Plus) 0,75MUI + 125mg x 20 Viên
```

Output:

```json
{
  "generic_text": "Spiramycin + Metronidazol",
  "brand_text": "Spirastad Plus",
  "strength_text": "0,75MUI + 125mg",
  "quantity": {
    "value": 20,
    "unit": "Viên"
  },
  "ingredients": [
    {
      "name": "Spiramycin",
      "strength_raw": "0,75MUI"
    },
    {
      "name": "Metronidazol",
      "strength_raw": "125mg"
    }
  ],
  "is_combination": true
}
```

## 10.5. IngredientEvidenceResolver

Chuẩn hóa tên hoạt chất trong đơn sang `evidence_slug`.

Các lớp xử lý:

- Exact match.
- No-diacritics match.
- Manual alias.
- Salt/form stripping.
- Prefix stripping.
- Fuzzy matching có safety guard.

Không được fuzzy-map các hoạt chất gần tên nhưng khác nhau về lâm sàng.

Ví dụ không được map sai:

```text
Omeprazole ≠ Esomeprazole
Levofloxacin ≠ Ofloxacin
Cefuroxime ≠ Cefixime
```

## 10.6. NormalizeDrugsService

Chuẩn hóa danh sách thuốc.

Thứ tự ưu tiên:

```text
1. Nếu có generic_text/ingredients → resolve trực tiếp qua IngredientEvidenceResolver
2. Nếu có brand_text → dùng brand để kiểm tra chéo
3. Nếu không có generic_text → dùng Long Châu fallback
4. Nếu product mapping thất bại → thử ingredient-only fallback
5. Nếu vẫn không xác định được → unmatched, yêu cầu bác sĩ/dược sĩ xác nhận
```

Các trạng thái chính:

```text
ingredient_with_brand
ingredient_only
product_matched
unmatched
```
human-in-the-loop
Nếu có thuốc unmatched hoặc brand_conflict:
→ hệ thống hiển thị danh sách cần xác nhận
→ bác sĩ chọn/nhập hoạt chất đúng
→ hệ thống lưu mapping đã xác nhận vào doctor_verified_mappings hoặc user_confirmed_mappings
→ lần sau có thể dùng mapping này để hỗ trợ chuẩn hóa

## 10.7. QdrantRetrieverService

Truy xuất evidence y khoa từ Qdrant.

Hệ thống không embedding nguyên đơn thuốc rồi tìm kiếm tự do trên toàn bộ kho dữ liệu.

Thay vào đó:

```text
extract hoạt chất / entity
→ chuẩn hóa sang evidence_slug
→ xác định loại truy vấn
→ tạo query text đã chuẩn hóa
→ embedding query
→ search Qdrant có filter theo slug và section
→ rerank theo section liên quan
→ lấy evidence chunks phù hợp
```

Ví dụ với câu hỏi:

```text
Omeprazol có tương tác với Clopidogrel không?
```

Hệ thống sẽ ưu tiên truy xuất các section như:

```text
tuong_tac_thuoc
than_trong
```

## 10.8. DoctorMemoryService

Phụ trách:

- Lưu note riêng của bác sĩ sau khi bác sĩ đọc báo cáo và chủ động ghi chú.
- Gắn note với `doctor_id`, danh sách hoạt chất, ngữ cảnh bệnh nhân và nội dung note.
- Khi có đơn thuốc/câu hỏi mới, truy xuất note theo `doctor_id` và các hoạt chất/ngữ cảnh liên quan.
- Chỉ trả về note nếu note có liên quan đến hoạt chất hoặc tình huống đang phân tích.
- Nếu không có note phù hợp, workflow tiếp tục chỉ với evidence y khoa chung.
- Đảm bảo bác sĩ không xem được note của người khác.

DoctorMemoryService không tự tạo kết luận lâm sàng. Service này chỉ cung cấp ghi chú cá nhân phù hợp để LLM xem như ngữ cảnh bổ sung, bên cạnh evidence y khoa đã được truy xuất.

## 10.9. LLM Analyzer

Sử dụng Gemini để phân tích nguy cơ dựa trên evidence đã truy xuất.

Nguyên tắc:

- Không để Gemini tự kết luận dựa trên kiến thức nội tại.
- Nếu evidence không đủ, phải trả lời chưa đủ thông tin.
- Mọi cảnh báo cần có nguồn.
- Không được tự ý kê đơn, chẩn đoán hoặc yêu cầu bệnh nhân ngừng thuốc.

## 10.10. Safety Layer

Kiểm tra đầu ra cuối cùng trước khi trả về người dùng.

Nhiệm vụ:

- Chặn câu hỏi ngoài phạm vi.
- Không tự chẩn đoán.
- Không tự kê đơn.
- Không khuyên bệnh nhân tự ngừng thuốc.
- Không kết luận an toàn nếu còn thuốc chưa xác định được hoạt chất.
- Luôn nhấn mạnh bác sĩ/dược sĩ là người quyết định cuối cùng.

## 11. Kiến trúc tổng thể

```text
Doctor / Pharmacist
        ↓
ReactJS Frontend
        ↓
FastAPI Backend
        ↓
Authentication Layer
        ↓
Intent Router
        ↓
Prescription Parser / MedicationLineParser
        ↓
NormalizeDrugsService
        ↓
IngredientEvidenceResolver
        ↓
QdrantRetrieverService
        ↓
DoctorMemoryService
        ↓
LangGraph Agent Workflow
        ↓
Gemini Risk Analysis / Report Generation
        ↓
Safety Layer
        ↓
Final Report
```

## 12. LangGraph Agent Workflow

Workflow kiểm tra đơn thuốc trong MVP:

```text
START
→ classify_intent
→ parse_prescription_text
→ parse_medication_lines
→ normalize_drugs
→ resolve_ingredients
→ retrieve_medical_evidence
→ retrieve_doctor_memory
→ analyze_risks
→ generate_report
→ safety_check
→ END
```

Workflow hỏi đáp đơn lẻ:

```text
START
→ classify_intent
→ extract_drug_entities
→ resolve_ingredients
→ retrieve_medical_evidence
→ retrieve_doctor_memory
→ generate_answer
→ safety_check
→ END
```

## 13. Công nghệ sử dụng

| Thành phần | Công nghệ | Vai trò |
|---|---|---|
| Frontend | ReactJS | Giao diện nhập đơn, hỏi đáp, xem báo cáo, lưu note |
| Backend | FastAPI | Xây API, xác thực, gọi workflow |
| Agent Workflow | LangGraph | Điều phối pipeline nhiều bước |
| LLM | Google Gemini API | Phân tích nguy cơ và sinh báo cáo dựa trên evidence |
| Vector Database | Qdrant | Lưu evidence y khoa và doctor memory |
| Embedding | intfloat/multilingual-e5-base | Vector hóa dữ liệu tiếng Việt/Anh |
| Database phụ | SQLite/PostgreSQL | Lưu user, lịch sử, mapping xác nhận |
| Authentication | JWT | Xác thực bác sĩ và phân quyền doctor memory |

## 14. Thiết kế lưu trữ

## 14.1. Qdrant collection: clinical_auditor_evidence

Lưu evidence y khoa dùng chung.

Payload mẫu:

```json
{
  "chunk_id": "trungtamthuoc:ingredient:paracetamol:tuong_tac_thuoc:001",
  "source": "trungtamthuoc",
  "source_name": "Trung Tâm Thuốc / Dược thư Quốc gia Việt Nam",
  "entity_type": "ingredient",
  "entity_name": "Paracetamol",
  "slug": "paracetamol",
  "section": "tuong_tac_thuoc",
  "url": "...",
  "text": "..."
}
```

## 14.2. Qdrant collection: doctor_memory

Lưu note riêng của từng bác sĩ. Collection này phục vụ semantic search trên ghi chú cá nhân, nhưng mọi truy vấn bắt buộc phải filter theo `doctor_id`.

Payload mẫu:

```json
{
  "note_id": "note_001",
  "doctor_id": "doctor_001",
  "active_ingredients": ["aspirin", "warfarin"],
  "patient_context": ["người cao tuổi", "nguy cơ chảy máu"],
  "note": "Với bệnh nhân cao tuổi dùng aspirin và warfarin, cần chú ý nguy cơ chảy máu.",
  "visibility": "private",
  "created_at": "2026-06-21"
}
```

Cơ chế truy xuất:

```text
resolved_ingredients + patient_context + doctor_id
→ embedding query cho memory
→ search doctor_memory với filter doctor_id
→ ưu tiên note có active_ingredients trùng hoặc gần với đơn/câu hỏi hiện tại
→ đưa note phù hợp vào LLM như personal clinical note
```

Nếu không tìm thấy note phù hợp, phần Doctor Memory sẽ để trống và hệ thống vẫn phân tích bình thường dựa trên evidence y khoa chung.

## 14.3. SQL database phụ

Dùng SQLite trong MVP, có thể mở rộng PostgreSQL sau.

Các bảng chính:

```text
users
prescription_history
report_history
drug_products
active_ingredients
user_confirmed_mappings
doctor_verified_mappings
```

## 15. Format báo cáo đầu ra

Báo cáo kiểm tra đơn thuốc gồm các phần:

## 15.1. Thông tin đơn thuốc

- Thông tin bệnh nhân nếu có.
- Chẩn đoán hoặc bệnh nền nếu có.
- Danh sách thuốc đã nhận diện.
- Hoạt chất đã chuẩn hóa.
- Tên thương mại nếu có.
- Hàm lượng, số lượng, cách dùng.

## 15.2. Tổng quan mức độ rủi ro

Các mức:

```text
LOW
MODERATE
HIGH
INSUFFICIENT_INFORMATION
```

Nếu còn thuốc chưa xác định được hoạt chất, hệ thống không được kết luận đơn thuốc là an toàn.

Kết luận phải là:

```text
Chưa đủ thông tin để kết luận toàn bộ đơn thuốc.
```

## 15.3. Cảnh báo thuốc - thuốc

Mỗi cảnh báo gồm:

- Cặp hoạt chất.
- Mức độ nguy cơ.
- Lý do.
- Evidence liên quan.
- Nguồn tham khảo.

## 15.4. Cảnh báo thuốc - bệnh nền

Mỗi cảnh báo gồm:

- Hoạt chất liên quan.
- Bệnh nền liên quan.
- Lý do.
- Evidence liên quan.
- Nguồn tham khảo.

## 15.5. Cảnh báo thuốc - thực phẩm/lối sống

Mỗi cảnh báo gồm:

- Hoạt chất liên quan.
- Thực phẩm/lối sống liên quan.
- Lý do.
- Evidence liên quan.
- Nguồn tham khảo.

## 15.6. Ghi chú riêng của bác sĩ

Hiển thị note liên quan từ Doctor Memory nếu có.

Phần này phải tách biệt với evidence y khoa chung và nên ghi rõ đây là ghi chú cá nhân của bác sĩ, không phải nguồn tham khảo y khoa chính thống.

Nếu không có note phù hợp với hoạt chất hoặc ngữ cảnh hiện tại, báo cáo không cần hiển thị phần này hoặc có thể hiển thị trạng thái:

```text
Không có ghi chú riêng liên quan.
```

## 15.7. Thông tin còn thiếu

Ví dụ:

- Dị ứng thuốc.
- Bệnh nền.
- Thai kỳ/cho con bú.
- Chức năng gan/thận.
- Thuốc đang dùng khác.
- Thuốc chưa xác định được hoạt chất.

## 15.8. Kết luận hỗ trợ

Ví dụ:

```text
Hệ thống chỉ hỗ trợ cảnh báo và tra cứu bằng chứng. Bác sĩ/dược sĩ là người quyết định cuối cùng.
```

## 16. API dự kiến

## 16.1. Authentication

```http
POST /auth/login
```

Đăng nhập bác sĩ và trả JWT token.

## 16.2. Kiểm tra đơn thuốc text

```http
POST /prescriptions/check-text
```

Input:

```json
{
  "prescription_text": "...",
  "question": "Kiểm tra tương tác thuốc trong đơn này"
}
```

Output:

```json
{
  "recognized_drugs": [],
  "risk_summary": {},
  "alerts": [],
  "missing_information": [],
  "sources": []
}
```

## 16.3. Hỏi đáp thuốc

```http
POST /chat/query
```

Input:

```json
{
  "message": "Aspirin có tương tác với Warfarin không?"
}
```

## 16.4. Lưu note bác sĩ

```http
POST /doctor-notes
```

Input:

```json
{
  "active_ingredients": ["aspirin", "warfarin"],
  "patient_context": ["người cao tuổi"],
  "note": "Cần chú ý nguy cơ xuất huyết."
}
```

## 16.5. Tìm note bác sĩ

```http
GET /doctor-notes/search
```

## 17. Giao diện MVP

Các màn hình chính:

## 17.1. Login

- Bác sĩ đăng nhập.
- Backend trả JWT.
- Frontend lưu token để gọi API.

## 17.2. Chat / Prescription Check

- Ô nhập đơn thuốc dạng text.
- Ô nhập yêu cầu kiểm tra.
- Nút gửi.
- Hiển thị quá trình xử lý nếu cần.

## 17.3. Report Result

Hiển thị:

- Danh sách thuốc đã nhận diện.
- Hoạt chất đã chuẩn hóa.
- Cảnh báo nguy cơ.
- Mức độ rủi ro.
- Evidence và nguồn.
- Thông tin còn thiếu.
- Kết luận hỗ trợ.

## 17.4. Doctor Memory

- Bác sĩ nhập note sau khi xem báo cáo.
- Hiển thị note liên quan nếu có.
- Note được gắn theo doctor_id.

## 18. Quy trình kiểm tra đơn thuốc trong MVP

```text
1. Bác sĩ đăng nhập.
2. Bác sĩ nhập đơn thuốc dạng text.
3. Backend nhận request.
4. Intent Router xác định đây là PRESCRIPTION_CHECK.
5. Prescription Parser tách thông tin đơn.
6. MedicationLineParser tách từng dòng thuốc.
7. NormalizeDrugsService chuẩn hóa thuốc.
8. IngredientEvidenceResolver resolve hoạt chất sang evidence_slug.
9. QdrantRetrieverService truy xuất evidence theo slug và section.
10. DoctorMemoryService tìm note riêng liên quan theo doctor_id, hoạt chất và ngữ cảnh bệnh nhân.
11. Nếu có note phù hợp, Gemini phân tích dựa trên evidence y khoa và note riêng của bác sĩ; nếu không có note, Gemini chỉ phân tích dựa trên evidence y khoa.
12. Report Generator tạo báo cáo.
13. Safety Layer kiểm tra đầu ra.
14. Frontend hiển thị báo cáo.
15. Bác sĩ có thể lưu note chuyên môn.
```

## 19. Quy trình hỏi đáp đơn lẻ

```text
1. Bác sĩ nhập câu hỏi.
2. Intent Router phân loại intent.
3. Hệ thống trích xuất hoạt chất/entity.
4. IngredientEvidenceResolver chuẩn hóa hoạt chất.
5. QdrantRetrieverService truy xuất evidence.
6. DoctorMemoryService lấy note riêng liên quan nếu có.
7. Gemini trả lời dựa trên evidence y khoa; note riêng chỉ đóng vai trò ngữ cảnh bổ sung khi phù hợp.
8. Safety Layer kiểm tra.
9. Trả câu trả lời có nguồn.
```

## 20. Nguyên tắc an toàn

Hệ thống phải tuân thủ các nguyên tắc:

1. Không thay thế bác sĩ/dược sĩ.
2. Không tự kê đơn.
3. Không tự chẩn đoán.
4. Không yêu cầu bệnh nhân tự ý ngừng thuốc.
5. Không trả lời chắc chắn nếu không có evidence.
6. Không tự đoán hoạt chất nếu không đủ cơ sở.
7. Không kết luận an toàn nếu có thuốc chưa xác định được hoạt chất.
8. Luôn hiển thị nguồn evidence.
9. Luôn tách biệt note cá nhân của bác sĩ với nguồn y khoa chung.
10. Luôn chặn câu hỏi ngoài phạm vi.

## 21. Đánh giá hệ thống

Các nhóm đánh giá đề xuất:

## 21.1. Đánh giá nhận diện và chuẩn hóa thuốc

- Parser có tách đúng hoạt chất không?
- Có nhận diện đúng tên thương mại không?
- Có xử lý thuốc phối hợp không?
- Có phát hiện thuốc không xác định được không?

## 21.2. Đánh giá retrieval

- Có truy xuất đúng hoạt chất không?
- Có truy xuất đúng section không?
- Có giảm nhiễu so với search toàn kho không?
- Có lấy được evidence phù hợp với câu hỏi không?

## 21.3. Đánh giá báo cáo

- Báo cáo có cảnh báo đúng trọng tâm không?
- Có nguồn tham khảo không?
- Có phân biệt mức độ nguy cơ không?
- Có nêu thông tin còn thiếu không?
- Có tránh kết luận quá mức không?

## 21.4. Đánh giá safety

- Có chặn câu hỏi ngoài phạm vi không?
- Có từ chối khi thiếu evidence không?
- Có tránh tự chẩn đoán/kê đơn không?
- Có nhấn mạnh vai trò quyết định cuối cùng của bác sĩ/dược sĩ không?

## 22. Kết quả kỳ vọng của đồ án

Sau khi hoàn thành, đồ án cần đạt được:

1. Một hệ thống backend FastAPI có thể nhận đơn thuốc text và trả báo cáo kiểm tra.
2. Một workflow AI Agent rõ ràng, có nhiều bước xử lý.
3. Một kho evidence y khoa lưu trong Qdrant.
4. Một pipeline RAG có filter theo hoạt chất và section.
5. Một module chuẩn hóa thuốc theo hướng generic-first.
6. Một module Doctor Memory theo doctor_id.
7. Một Safety Layer giới hạn phạm vi trả lời.
8. Một giao diện React đơn giản để demo.
9. Một bộ test case để đánh giá parser, retrieval và báo cáo.
10. Một báo cáo đồ án giải thích rõ cơ sở lý thuyết, thiết kế, triển khai và đánh giá.

## 23. Hướng phát triển sau đồ án

Sau MVP, hệ thống có thể mở rộng:

1. Thêm OCR/upload ảnh/PDF đơn thuốc.
2. Cho phép bác sĩ xác nhận lại JSON sau OCR.
3. Tăng chất lượng drug normalization bằng alias dictionary lớn hơn.
4. Mở rộng nguồn dữ liệu chính thống.
5. Thêm đánh giá mức độ tương tác theo guideline rõ hơn.
6. Tối ưu Qdrant retrieval và reranking.
7. Triển khai production với PostgreSQL, Docker và CI/CD.
8. Bổ sung phân quyền nhiều vai trò: bác sĩ, dược sĩ, admin.
9. Bổ sung audit log cho các lần kiểm tra đơn.
10. Tối ưu caching embedding model và retriever service.

## 24. Kết luận

Clinical Auditor AI Agent là một hệ thống hỗ trợ kiểm tra đơn thuốc dựa trên AI Agent, RAG và Doctor Memory. Điểm cốt lõi của hệ thống không nằm ở việc hỏi đáp tài liệu đơn thuần, mà ở việc xây dựng một quy trình kiểm tra đơn thuốc có kiểm soát: nhận đơn, chuẩn hóa hoạt chất, truy xuất bằng chứng, phân tích nguy cơ, kiểm tra an toàn và sinh báo cáo có nguồn.

Trong phạm vi đồ án tốt nghiệp, hệ thống tập trung vào đầu vào text thay vì OCR để đảm bảo tính khả thi, dễ kiểm thử và phù hợp với năng lực triển khai của sinh viên. OCR/upload PDF được xem là hướng mở rộng sau khi pipeline AI Agent và RAG đã ổn định.

Hệ thống không thay thế bác sĩ/dược sĩ, mà đóng vai trò công cụ hỗ trợ tra cứu, cảnh báo và ra quyết định dựa trên bằng chứng.
