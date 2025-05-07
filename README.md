Một ứng dụng mô phỏng bằng Python sử dụng **Tkinter** để trình bày trực quan từng bước của quá trình **nén ảnh bằng LDA (Latent Dirichlet Allocation)** — từ ảnh gốc đến phục hồi ảnh mờ dựa trên phân tích chủ đề.

## Mô tả chức năng

Ứng dụng thực hiện:

1. **Chọn ảnh từ máy tính** (`.jpg`, `.jpeg`, `.png`)
2. Chuyển ảnh thành **xám** và **resize** thành 128x128
3. **Cắt ảnh** thành các patch 8x8
4. Gán các patch thành các **từ** bằng **KMeans clustering**
5. Tạo **corpus** từ các từ và áp dụng **LDA** để phân tích chủ đề
6. Hiển thị **biểu đồ phân bố chủ đề**
7. **Phục hồi ảnh** từ topic dominant bằng cách tính trung bình các patch

## Công nghệ sử dụng

- `Tkinter` – Tạo giao diện người dùng
- `Pillow` – Xử lý ảnh
- `numpy`, `matplotlib` – Tính toán và biểu đồ
- `sklearn.cluster.KMeans` – Clustering
- `gensim.models.LdaModel` – LDA topic modeling
- `matplotlib.backends.backend_tkagg.FigureCanvasTkAgg` – Hiển thị biểu đồ trong Tkinter

## Cài đặt

### 1. Clone dự án

```bash
git clone https://github.com/your_username/image-compression-lda-tkinter.git
cd image-compression-lda-tkinter

