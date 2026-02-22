<div align="center">
  <img src="./assets/banner1.jpg" width="100%">
</div>

# 🚀 DỰ ĐOÁN HÀNH VI NGƯỜI DÙNG (USER BEHAVIOR PREDICTION)

Một doanh nghiệp lớn đang đối mặt với bài toán tối ưu hóa chi phí vận hành kho bãi. Để giải quyết vấn đề này, doanh nghiệp mong muốn dự đoán trước các hành vi mua sắm của khách hàng dựa trên chuỗi hành động tương tác của họ trong quá khứ, từ đó giúp đội ngũ Sales phân tích và có chiến lược nhập hàng, quản lý tồn kho hiệu quả.

Thí sinh, với vai trò một Data Scientist, cần giải quyết bài toán **Dự đoán hành vi người dùng**.

---

# Quick Start

Truy cập và tham gia cuộc thi trên hệ thống Kaggle tại đường link sau: 
[DỰ ĐOÁN HÀNH VI NGƯỜI DÙNG (USER BEHAVIOR PREDICTION) trên Kaggle](https://www.kaggle.com/t/bb073834e2f240be99c0b7e4672d96da)

---

# Nội dung
- [Mô tả bài toán](#mô-tả-bài-toán)
    - [1. Đặc điểm bài toán](#1-đặc-điểm-bài-toán)
    - [2. Cấu trúc dữ liệu](#2-cấu-trúc-dữ-liệu)
    - [3. Kết quả dự đoán](#3-kết-quả-dự-đoán)
    - [4. Chỉ số đánh giá](#4-chỉ-số-đánh-giá)
- [Starter Kit](#starter-kit)
- [Các mốc thời gian quan trọng](#các-mốc-thời-gian-quan-trọng)
- [Quy định cuộc thi](#quy-định-cuộc-thi)
- [Tài nguyên](#tài-nguyên)

---

# Mô tả bài toán

**Nhiệm vụ:**
Phân tích dữ liệu và xây dựng mô hình phân loại đa đầu ra (Multi-output Classifier) để dự đoán 6 thuộc tính hành vi độc lập của một khách hàng trong tương lai, dựa trên chuỗi hành động trong quá khứ của họ.

### 1. Đặc điểm bài toán

- **Dữ liệu đầu vào (Input)** là một chuỗi các hành động liên tiếp của khách hàng theo thời gian. Trong vòng 4 tuần, khách hàng có thể cập nhật giao dịch và thay đổi nội dung đơn hàng. Toàn bộ các hành động này đã được mã hóa thành các con số định danh nhằm bảo mật thông tin.
- **Dữ liệu đầu ra (Output)** gồm 6 biến mục tiêu độc lập nhằm hỗ trợ doanh nghiệp xác định xem mỗi ngày cần nhập và phân bổ những loại hàng hóa nào, từ đó tối ưu chi phí lưu kho và vận hành. Sáu biến này đại diện cho 6 khía cạnh hành vi khác nhau của người dùng và không có tính liên quan trực tiếp đến nhau. 

### 2. Cấu trúc dữ liệu

Dữ liệu đã được Ban tổ chức chia sẵn thành 3 tập độc lập: Tập huấn luyện (Train), Tập kiểm định (Validation), và Tập kiểm thử (Test), tương ứng với các tập dữ liệu **X_train.csv và Y_train.csv**, **X_val.csv và Y_val.csv**, **X_test.csv**. 

Mỗi dòng trong **tập X (Đầu vào)** sẽ tương ứng trực tiếp với một dòng ở **tập Y (Đầu ra)**, bao gồm 6 thuộc tính cần dự đoán. Mỗi dòng đại diện cho một phiên giao dịch khách hàng duy nhất.

- **Tập X (Đầu vào):** Chứa chuỗi các hành động của người dùng đã được mã hóa. Lưu ý: Độ dài chuỗi đầu vào của mỗi khách hàng là khác nhau (Variable-length sequence) do tần suất và số lượng tương tác của mỗi người dùng không đồng nhất.
- **Tập Y (Đầu ra):** Chứa 6 cột dữ liệu, tương ứng với 6 thuộc tính hành vi cần dự đoán. Dữ liệu đầu ra yêu cầu định dạng số nguyên không dấu 16-bit (UINT16). Nếu mô hình dự đoán ra số thực (float), thí sinh cần ép kiểu kết quả về UINT16.

> **LƯU Ý:**
> Để đảm bảo tính công bằng trong học thuật và tối ưu hiệu năng trên bảng xếp hạng, Ban tổ chức quy định rõ cách thức sử dụng các tập dữ liệu như sau:
> 1. **Giai đoạn Nghiên cứu, EDA và Đánh giá nội bộ (Bắt buộc cô lập):** Trong quá trình Phân tích Dữ liệu Khám phá (EDA), thiết lập quy trình Tiền xử lý (Feature Engineering) và tính toán các chỉ số (metrics) để báo cáo, thí sinh tuyệt đối không được gộp tập Train và Validation.
> 2. **Giai đoạn Tối ưu hóa điểm số trên Kaggle (Được phép gộp):** Sau khi đã chốt được kiến trúc mô hình và các tham số tối ưu (Hyperparameters) thông qua quá trình đánh giá ở Giai đoạn 1, để tạo ra file kết quả (Submission) nộp lên hệ thống Kaggle, thí sinh **ĐƯỢC PHÉP gộp chung** tập Train và Validation lại thành một tập dữ liệu lớn hơn. Các bạn có thể chạy lại toàn bộ quy trình tiền xử lý và huấn luyện (Retrain) trên tập dữ liệu tổng hợp này để mô hình học được nhiều thông tin nhất có thể trước khi dự đoán trên tập Test cuối cùng.

### 3. Kết quả dự đoán

Với mỗi `ID` trong tập X_test.csv, thí sinh cần dự báo 6 thuộc tính tương ứng. File nộp cần có định dạng `.csv` với tiêu đề cột chính xác như sau:
```csv
id,attr_1,attr_2,attr_3,attr_4,attr_5,attr_6
gpbfd,0,0,0,0,0,0
w22ee,0,0,0,0,0,0
wyw95,0,0,0,0,0,0
izx4w,0,0,0,0,0,0
c6o2d,0,0,0,0,0,0
```

### 4. Chỉ số đánh giá

Cuộc thi sử dụng chỉ số **Exact-Match Accuracy (Độ chính xác khớp tuyệt đối)** trên tập Test để đánh giá xếp hạng.

**Quy tắc tính điểm:**
Bài toán yêu cầu dự đoán một vector đầu ra gồm 6 chiều (tương ứng với 6 thuộc tính hành vi) cho mỗi người dùng.

**Một dự đoán chỉ được tính là Chính xác khi và chỉ khi mô hình dự đoán đúng đồng thời cả 6 giá trị của vector đầu ra**. Nếu mô hình dự đoán sai dù chỉ một trong 6 chiều, toàn bộ kết quả dự đoán của dòng dữ liệu đó sẽ bị đánh giá là Không chính xác.

**Công thức chỉ số Exact-Match Accuracy**

$$\text{Accuracy} = \frac{N_{acc}}{N}$$

**Trong đó:**
- $N_{acc}$: Số lượng mẫu dự đoán đúng hoàn toàn (khớp đồng thời cả 6 thuộc tính).
- $N$: Tổng số lượng mẫu trên tập Test.

---

# Starter Kit

Chúng tôi cung cấp **starter kit** cho bài toán để giúp người tham gia bắt đầu dễ dàng hơn:
- **Starter Kit**: [`starter_kit/`](./starter_kit/)

Và các công cụ hỗ trợ:
- **Script đánh giá**: [`evaluation_script/`](./evaluation_script/)
- **File nộp mẫu**: [`sample_submission_file/`](./sample_submission_file/)
- **Dữ liệu cuộc thi**: [`dataset/`](./dataset/) *(Lưu ý: Thư mục này được thiết lập bỏ qua trong `.gitignore`. Thí sinh cần tự tải dữ liệu từ nền tảng cuộc thi trên Kaggle và đặt vào thư mục này)*

---

# Các mốc thời gian quan trọng

| Mô tả                         | Thời gian               |
|-------------------------------|------------------------|
| ~~Mở đơn đăng ký tham gia~~    | ~~01/01/2026 - 15/01/2026~~   |
| ~~Vòng loại~~    |  ~~21/01/2026 - 12/02/2026~~           |
| ~~Báo cáo vòng loại~~    |  ~~13/02/2026~~      |
| Vòng bán kết      | 23/02/2026 - 07/03/2026    |
| Báo cáo vòng bán kết      | 09/03/2026    |
| Vòng chung kết        | 12/03/2026 - 19/03/2026    |
| Báo cáo vòng chung kết   | 21/03/2026       |

---

# Quy định cuộc thi

### PHẦN 1: QUY ĐỊNH VỀ ĐẠO ĐỨC VÀ ỨNG XỬ
1. **Văn hóa tôn trọng:**
- Thí sinh phải giữ thái độ tôn trọng, đúng mực, hợp tác với Ban Tổ Chức (BTC), Ban Giám Khảo (BGK), Mentor và các đội thi khác.
- Nghiêm cấm mọi hành vi quấy rối, công kích cá nhân, phân biệt vùng miền/giới tính, hoặc đe dọa vũ lực (dù là trực tiếp hay trên không gian mạng).
2. **Phát ngôn trên mạng xã hội:**
- Mọi thắc mắc, khiếu nại cần được gửi qua kênh email chính thức hoặc hotline của BTC. Nghiêm cấm việc đăng tải thông tin sai lệch, chưa kiểm chứng hoặc sử dụng ngôn từ kích động nhằm hạ thấp uy tín cuộc thi trên mạng xã hội.
3. **Tính trung lập:**
- Nghiêm cấm các hành vi tặng quà, hối lộ hoặc lợi dụng mối quan hệ cá nhân với thành viên BTC/BGK dẫn đến xung đột lợi ích, ảnh hưởng đến kết quả trung thực của cuộc thi.

### PHẦN 2: QUY ĐỊNH VỀ LIÊM CHÍNH HỌC THUẬT & CHUYÊN MÔN
4. **Bản quyền và Ý tưởng:**
- Sản phẩm dự thi phải thuộc quyền sở hữu của đội thi. Nghiêm cấm mọi hành vi sao chép, đạo nhái ý tưởng từ các đội khác hoặc các nguồn có sẵn mà không trích dẫn hợp lệ.
- Nếu phát hiện gian lận/ăn cắp ý tưởng (kể cả sau khi trao giải), BTC có quyền thu hồi giải thưởng và công bố công khai hành vi vi phạm.
- Sản phẩm thuộc quyền sở hữu trí tuệ của đội thi. Tuy nhiên, BTC có quyền sử dụng hình ảnh, mô tả sản phẩm cho mục đích truyền thông phi thương mại và lưu trữ hồ sơ cuộc thi.
5. **Giới hạn sự hỗ trợ của Mentor:**
- Mentor chỉ đóng vai trò định hướng, gợi ý phương pháp tư duy.
- Nghiêm cấm nhờ Mentor trực tiếp viết code, thiết kế slide, hoặc tham gia vào quá trình xây dựng sản phẩm dự thi. Nếu phát hiện, cả đội thi sẽ bị hủy bỏ kết quả và tư cách dự thi.
6. **Sử dụng danh nghĩa BTC:** 
- Thí sinh không được phép mạo danh BTC để tạo fanpage, group, lừa đảo thông tin, tài chính của các thí sinh khác.

### PHẦN 3: QUY ĐỊNH VỀ CƠ CẤU ĐỘI THI & HỒ SƠ
7. **Tính ổn định nhân sự:**
- Danh sách thành viên chốt tại vòng loại là danh sách cuối cùng. Không chấp nhận việc ghép đội, thay thế thành viên hoặc nhờ người thi hộ dưới mọi hình thức.
- Trong trường hợp bất khả kháng (thành viên bỏ thi), đội thi vẫn phải tiếp tục với số lượng thành viên còn lại (miễn là đáp ứng mức tối thiểu: 02 thành viên).
8. **Trung thực về thông tin cá nhân:**
- Thí sinh chịu trách nhiệm hoàn toàn về tính chính xác của thông tin cá nhân (Thẻ sinh viên, CCCD). Nếu phát hiện khai gian, đội sẽ bị hủy bỏ kết quả thi và bị loại ngay lập tức.

### PHẦN 4: QUY ĐỊNH VỀ VẬN HÀNH & HẬU CẦN
9. **Trách nhiệm cập nhật thông tin:**
- BTC sẽ gửi thông báo qua Email đăng ký và Group Zalo chính thức. Thí sinh có trách nhiệm kiểm tra hộp thư (bao gồm cả Spam) hàng ngày. BTC không chịu trách nhiệm nếu thí sinh bỏ lỡ thông tin.
10. **Quy định nộp bài:**
- Hệ thống đóng nộp bài đúng giờ quy định. Mọi lý do cá nhân (mất mạng, lỗi máy tính,...) hoặc nộp muộn đều không được chấp nhận (trừ lỗi server từ phía BTC).
- BTC không chấp nhận chỉnh sửa/thay thế file nộp bài sau khi hết hạn. Trong trường hợp hệ thống gặp sự cố, thí sinh phải gửi bài qua Email BTC trước giờ deadline để làm bằng chứng.
- Đội thi phải nộp đầy đủ các tài liệu liên quan phục vụ cho việc báo cáo. Nếu không nộp đủ, đội thi sẽ không được tham gia báo cáo và bị loại trực tiếp.
11. **Bảo quản tài sản & Hình ảnh:**
- Tại các buổi Offline, thí sinh làm hư hỏng thiết bị của BTC do cố ý hoặc bất cẩn sẽ phải bồi thường theo giá trị thị trường tại thời điểm xảy ra sự cố hoặc theo báo giá sửa chữa từ đơn vị cung cấp dịch vụ.
- Thí sinh cam kết cung cấp thông tin cá nhân để BTC, Nhà tài trợ & Nhà bảo trợ chính thức của cuộc thi phục vụ công tác tổ chức, trao giải trong và sau cuộc thi.
- BTC có quyền sử dụng hình ảnh thí sinh trong quá trình diễn ra cuộc thi cho mục đích truyền thông phi thương mại.
- Yêu cầu trang phục lịch sự, phù hợp thuần phong mỹ tục tại các buổi trình bày/Offline.

### PHẦN 5: QUY ĐỊNH VỀ KHIẾU NẠI VÀ GIẢI QUYẾT TRANH CHẤP
- Mọi khiếu nại về kết quả hoặc vi phạm quy chế phải được gửi trực tiếp từ email của Đội trưởng đến email chính thức của BTC trong vòng 24 giờ kể từ khi sự việc phát sinh hoặc kết quả được công bố. BTC từ chối giải quyết các khiếu nại quá hạn, nặc danh, gửi sai kênh hoặc đăng tải công khai trên mạng xã hội khi chưa có kết luận chính thức.
- Người khiếu nại có nghĩa vụ cung cấp bằng chứng xác thực (hình ảnh, video, log file,...) chứng minh cho nội dung khiếu nại. BTC chỉ xem xét các sai sót về kỹ thuật, quy trình hoặc hành vi gian lận; không giải quyết khiếu nại liên quan đến đánh giá chuyên môn và ý kiến đánh giá chủ quan từ phía thí sinh.
- Hội đồng sẽ phản hồi khiếu nại trong 48 giờ và quyết định của BTC là cuối cùng. Các hành vi khiếu nại sai sự thật hoặc bôi nhọ cuộc thi sẽ bị tước danh hiệu và tư cách tham gia cuộc thi.
- Trong mọi trường hợp tranh chấp phát sinh chưa được quy định rõ trong quy định này, sẽ được BTC thảo luận và đưa ra phương án kịp thời.

---

# Tài nguyên

- [Script đánh giá](./evaluation_script/README.md)
- [Thư mục dữ liệu](./dataset/README.md)
- [Thư mục nộp mẫu (Sample Submission)](./sample_submission_file/)
- [Starter Kit](./starter_kit/)
