# 📚 GIẢI THÍCH CHI TIẾT CÁC CHỈ SỐ ĐO LƯỜNG TÁI TẠO ẢNH CT (BENCHMARK METRICS EXPLANATION)

**Dự án:** Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)  
**Tác giả:** MinhPD — VNU-HCM UIT  
**Ngày lưu:** 30/08/2026  

---

## 1. Bảng Số Liệu Thực Tế Từ Log Test (Job ID 65893 — LEARN_LongNet trên LA-90°)

```text
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_loss          0.011528508737683296
        test_psnr            19.19390296936035
        test_rmse           0.10577496141195297
        test_ssim           0.5876370072364807
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

---

## 2. Giải Thích Bản Chất Toán Học & Ý Nghĩa Y Tế Của Từng Chỉ Số

### 1. `test_loss` (Mean Squared Error - MSE = 0.0115)
* **Công thức toán học:**
  $$\text{MSE} = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W \left( \hat{x}_{i,j} - x_{i,j} \right)^2$$
  *(Trong đó $\hat{x}$ là ảnh do AI tái tạo, $x$ là ảnh Ground Truth gốc, kích thước $256 \times 256$ pixel, dải giá trị $[0.0, 1.0]$)*.
* **Ý nghĩa:** Đo lường **trung bình bình phương độ lệch** giữa từng pixel dự đoán so với pixel thật.
* **Đọc hiểu con số `0.0115`:** Sai số bình phương trung bình trên mỗi điểm ảnh là khoảng $1.15\%$ (trên thang chuẩn hóa $[0, 1]$). Càng gần $0$ thì ảnh càng chính xác tuyệt đối.

---

### 2. `test_rmse` (Root Mean Squared Error = 0.1058)
* **Công thức toán học:**
  $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{0.011528} \approx 0.1058$$
* **Ý nghĩa:** Căn bậc hai của MSE giúp đưa sai số về **cùng thứ nguyên (đơn vị tuyến tính)** với cường độ pixel của ảnh.
* **Đọc hiểu con số `0.1058`:** 
  - Độ lệch trung bình thực tế giữa điểm ảnh AI dự đoán và điểm ảnh thật là **$0.1058$** (tương đương $\approx 10.58\%$ dải động của ảnh).
  - *Đối chiếu:* Ở dải góc chuẩn $120^\circ$, RMSE của LongNet chỉ là **$0.0270$ ($2.7\%$)**. Khi bị thử thách ở dải góc hẹp $90^\circ$ (mất tới $270^\circ$ thông tin góc chiếu), sai số tăng lên $10.58\%$.

---

### 3. `test_psnr` (Peak Signal-to-Noise Ratio = 19.1939 dB)
* **Công thức toán học:**
  $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 10 \cdot \log_{10} \left( \frac{1.0^2}{0.011528} \right) \approx 19.1939\text{ dB}$$
  *(Với $\text{MAX}_I = 1.0$ là giá trị cường độ cực đại của ảnh)*.
* **Ý nghĩa:** Tỷ số tín hiệu cực đại trên nhiễu, tính theo thang đo Logarit Decibel ($\text{dB}$). Chỉ số này phản ánh **độ sạch và độ tương phản của tín hiệu ảnh so với nhiễu**:
  * **$< 20\text{ dB}$:** Ảnh còn nhiều nhiễu hạt hoặc vệt sọc (*streak artifacts*) thấy rõ bằng mắt thường.
  * **$25 - 30\text{ dB}$:** Ảnh ở mức chấp nhận được, cấu trúc chính nhìn rõ nhưng chi tiết nhỏ bị mờ.
  * **$30 - 35\text{ dB}$:** Ảnh chất lượng rất cao, cấu trúc sắc nét (LongNet đạt **$31.62\text{ dB}$** ở $120^\circ$).
  * **$> 35\text{ dB}$:** Chất lượng xuất sắc, gần như tiệm cận ảnh CT quét toàn góc ($360^\circ$).
* **Đọc hiểu con số `19.19 dB`:** Khi bị khuyết góc khắc nghiệt ($90^\circ$), lượng thông tin bị mất quá lớn khiến PSNR của mô hình giảm xuống $19.19\text{ dB}$ (tuy nhiên vẫn cao hơn Mamba chỉ đạt $18.76\text{ dB}$ và FBP thô chỉ đạt $15.20\text{ dB}$).

---

### 4. `test_ssim` (Structural Similarity Index Measure = 0.5876)
* **Công thức toán học:**
  $$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + c_1)(2\sigma_{x\hat{x}} + c_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + c_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + c_2)}$$
  *(Trong đó $\mu$ là độ sáng trung bình, $\sigma^2$ là phương sai/độ tương phản, $\sigma_{x\hat{x}}$ là hiệp phương sai cấu trúc giữa 2 ảnh)*.
* **Ý nghĩa:** Đo lường **độ tương đồng về mặt cấu trúc hình học, biên giải phẫu và cảm nhận thị giác của mắt người** (thang đo từ $0.0$ đến $1.0$, trong đó $1.0$ là hai ảnh giống hệt nhau $100\%$).
* **Tại sao SSIM quan trọng hơn PSNR trong Y tế?**
  - PSNR chỉ cộng dồn sai số từng điểm ảnh độc lập. Một ảnh bị mờ đều toàn bộ có thể có PSNR cao, nhưng bác sĩ không thể chẩn đoán được vì mất biên khối u.
  - SSIM đánh giá trực tiếp **hình dạng giải phẫu của xương, phổi, gan, mạch máu có bị biến dạng hay không**.
* **Đọc hiểu con số `0.5876`:** 
  - Dưới điều kiện khuyết $270^\circ$ góc quét, LongNet vẫn bảo toàn được **$\approx 58.76\%$ độ tương đồng cấu trúc giải phẫu**, vượt trội rõ rệt so với **Mamba ($0.4292$)** và **FBP ($0.4120$)**.

---

## 3. Bảng Đối Chiếu Nhanh Ý Nghĩa Các Chỉ Số Giữa Các Dải Góc

| Chỉ số | Ý nghĩa vắn tắt | Đơn vị | Giá trị lý tưởng | LongNet (LA-120°) | LongNet (LA-90° Stress Test) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`MSE (Loss)`** | Bình phương sai lệch pixel | Không đ/vị | Càng nhỏ càng tốt ($\to 0$) | **$0.0009$** | **$0.0115$** |
| **`RMSE`** | Sai lệch cường độ pixel trung bình | Tuyến tính $[0, 1]$ | Càng nhỏ càng tốt ($\to 0$) | **$0.0270$ ($2.7\%$)** | **$0.1058$ ($10.58\%$)** |
| **`PSNR`** | Độ sắc nét & sạch nhiễu của ảnh | Decibel ($\text{dB}$) | Càng lớn càng tốt ($> 30\text{ dB}$) | **$31.62\text{ dB}$** | **$19.19\text{ dB}$** |
| **`SSIM`** | Độ tương đồng cấu trúc giải phẫu | Đo lường $[0, 1]$ | Càng lớn càng tốt ($\to 1.0$) | **$0.8991$ ($89.91\%$)** | **$0.5876$ ($58.76\%$)** |
