import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim import corpora, models
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Biến toàn cục
steps = []
current_step = 0
img_path = None
img_gray = None
patches = None
words = None
dictionary = None
corpus = None
lda_model = None
restored_image = None

def select_image():
    global img_path, current_step
    file = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    if file:
        img_path = file
        current_step = 0
        steps.clear()
        define_steps()
        show_step()

def define_steps():
    steps.append(step_show_original)
    steps.append(step_gray_resize)
    steps.append(step_split_patches)
    steps.append(step_kmeans)
    steps.append(step_lda)
    steps.append(step_show_topic)
    steps.append(step_restore_image)  # Bước phục hồi ảnh

def show_step():
    if current_step < len(steps):
        steps[current_step]()
    else:
        status_label.config(text="✅ Hoàn tất quá trình nén và giải nén ảnh bằng LDA.")
        step_label.config(text="Hoàn thành")

def next_step():
    global current_step
    if current_step < len(steps) - 1:
        current_step += 1
        show_step()

def step_show_original():
    img = Image.open(img_path)
    show_image(img)
    status_label.config(text="Bước 1: Ảnh gốc")
    step_label.config(text="Bước 1: Ảnh gốc")

def step_gray_resize():
    global img_gray
    img = Image.open(img_path).convert("L").resize((128, 128))
    img_gray = np.array(img)
    show_image(Image.fromarray(img_gray))
    status_label.config(text="Bước 2: Chuyển ảnh sang xám và resize 128x128")
    step_label.config(text="Bước 2: Chuyển ảnh sang xám và resize 128x128")

def step_split_patches():
    global patches
    patch_size = 8
    h, w = img_gray.shape
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = img_gray[i:i+patch_size, j:j+patch_size].flatten()
            if patch.size == patch_size * patch_size:
                patches.append(patch)
    patches = np.array(patches)
    status_label.config(text=f"Bước 3: Chia ảnh thành {len(patches)} patch (8x8)")
    step_label.config(text="Bước 3: Chia ảnh thành {len(patches)} patch (8x8)")

def step_kmeans():
    global words
    kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
    kmeans.fit(patches)
    words = kmeans.predict(patches)
    status_label.config(text="Bước 4: Gán mỗi patch thành 1 từ bằng KMeans")
    step_label.config(text="Bước 4: Gán mỗi patch thành từ 1 bằng KMeans")

def step_lda():
    global dictionary, corpus, lda_model
    doc = [str(w) for w in words]
    dictionary = corpora.Dictionary([doc])
    corpus = [dictionary.doc2bow(doc)]
    lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42)
    status_label.config(text="Bước 5: Áp dụng LDA để nén ảnh")
    step_label.config(text="Bước 5: Áp dụng LDA để nén ảnh")

def step_show_topic():
    topic_dist = lda_model.get_document_topics(corpus[0], minimum_probability=0)
    topics = [f"Topic {i}" for i, _ in topic_dist]
    values = [v for _, v in topic_dist]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(topics, values, color='skyblue')
    ax.set_title("Bước 6: Biểu đồ phân bố chủ đề (LDA nén ảnh)")
    ax.set_ylabel("Xác suất")

    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    status_label.config(text="Bước 6: Biểu đồ phân bố chủ đề")
    step_label.config(text="Bước 6: Biểu đồ phân bố chủ đề")

def step_restore_image():
    global restored_image
    patch_size = 8
    # Lấy chủ đề của ảnh
    topic_dist = lda_model.get_document_topics(corpus[0], minimum_probability=0)
    dominant_topic = np.argmax([v for _, v in topic_dist])
    
    # Phân loại lại từng patch theo topic đã học từ LDA
    patch_topics = []
    for word in words:
        bow = dictionary.doc2bow([str(word)])
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
        patch_topics.append(np.argmax([v for _, v in topic_probs]))  # Chọn chủ đề có xác suất cao nhất
    patch_topics = np.array(patch_topics)

    # Trung bình patch theo topic
    num_topics = lda_model.num_topics
    avg_patches = np.zeros((num_topics, patches.shape[1]))
    for t in range(num_topics):
        indices = np.where(patch_topics == t)[0]
        if len(indices) > 0:
            avg_patches[t] = np.mean(patches[indices], axis=0)  # Tính trung bình của các patch thuộc cùng một chủ đề

    # Phục hồi patch theo topic đã phân
    restored_patches = avg_patches[patch_topics]  # Ghép các patch theo topic

    # Ghép lại ảnh từ các patch đã phục hồi
    h, w = img_gray.shape
    restored_image = np.zeros_like(img_gray)
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            restored_image[i:i+patch_size, j:j+patch_size] = restored_patches[idx].reshape(patch_size, patch_size)
            idx += 1

    restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)  # Cắt giá trị pixel trong khoảng 0-255
    
    # Hiển thị ảnh phục hồi
    show_image(Image.fromarray(restored_image))
    status_label.config(text="Bước 7: Phục hồi ảnh mờ từ topic dominant")
    step_label.config(text="Bước 7: Phục hồi ảnh mờ từ topic dominant")

def show_image(img):
    max_width, max_height = 300, 300  # Kích thước tối đa cho ảnh
    img.thumbnail((max_width, max_height))  # Giảm kích thước ảnh nếu cần
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# === Giao diện ===
# Tạo cửa sổ chính
root = tk.Tk()
root.title("📦 Mô phỏng nén ảnh bằng LDA - từng bước")
root.geometry("500x650")
root.configure(bg="#1e1e2f")

# Font chữ chung
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_BUTTON = ("Segoe UI", 12, "bold")
FONT_STATUS = ("Segoe UI", 11)
FONT_IMAGE_LABEL = ("Segoe UI", 10, "italic")

# Tiêu đề đẹp mắt
title_label = tk.Label(root, text="📦 Mô phỏng nén ảnh bằng LDA", 
                       font=FONT_TITLE, fg="#00b0f4", bg="#1e1e2f")
title_label.pack(pady=(30, 15))

# Nhãn hiển thị trạng thái bước đang làm
step_label = tk.Label(root, text="Bước 1: Ảnh gốc", font=("Segoe UI", 12), fg="#f1f1f1", bg="#1e1e2f")
step_label.pack(pady=10)

# Khung chứa nút chọn ảnh và nút Tiếp theo
button_frame = tk.Frame(root, bg="#1e1e2f")
button_frame.pack(pady=10)

btn_select = ttk.Button(button_frame, text="📂 Chọn ảnh", command=select_image)
btn_select.grid(row=0, column=0, padx=10)

btn_next = ttk.Button(button_frame, text="➡️ Tiếp theo", command=next_step)
btn_next.grid(row=0, column=1, padx=10)

# Khung hiển thị ảnh
image_frame = tk.Frame(root, width=320, height=320, bg="#292d3e", relief="ridge", borderwidth=4)
image_frame.pack(pady=20)
image_frame.pack_propagate(False)

image_label = tk.Label(image_frame, bg="#292d3e")
image_label.pack(expand=True)

img_desc_label = tk.Label(root, text="Ảnh hiển thị sẽ hiện tại đây", 
                          font=FONT_IMAGE_LABEL, fg="#a0a0a0", bg="#1e1e2f")
img_desc_label.pack()

# Khung hiển thị biểu đồ (Matplotlib canvas sẽ đặt ở đây)
plot_frame = tk.Frame(root, width=420, height=300, bg="#292d3e", relief="ridge", borderwidth=3)
plot_frame.pack(pady=20)
plot_frame.pack_propagate(False)

plot_label = tk.Label(plot_frame, text="Biểu đồ phân bố chủ đề sẽ hiện tại đây", 
                      font=FONT_IMAGE_LABEL, fg="#a0a0a0", bg="#292d3e")
plot_label.pack(expand=True)

# Thanh trạng thái đẹp
status_label = tk.Label(root, text="📌 Vui lòng chọn ảnh để bắt đầu", 
                        font=FONT_STATUS, fg="#00f6ff", bg="#1e1e2f", anchor="center")
status_label.pack(fill="x", pady=15, ipady=5)

# Tùy chỉnh style cho ttk.Button
style = ttk.Style()
style.theme_use('clam')
style.configure("TButton",
                font=FONT_BUTTON,
                foreground="#ffffff",
                background="#005c99",
                padding=10,
                relief="flat",  # Giảm khung viền
                borderwidth=0)  # Không có viền

# Thêm hiệu ứng hover và bấm
def on_hover_in(event):
    event.widget.config(background="#007acc", relief="solid", borderwidth=2)  # Khi hover vào nút
    event.widget.config(foreground="#ffffff")  # Đổi màu chữ

def on_hover_out(event):
    event.widget.config(background="#005c99", relief="flat", borderwidth=0)  # Khi hover ra ngoài
    event.widget.config(foreground="#ffffff")  # Đổi lại màu chữ khi không hover

def on_button_press(event):
    event.widget.config(background="#003f66", relief="sunken", borderwidth=2)  # Khi bấm nút
    event.widget.config(foreground="#ffffff")  # Đổi màu chữ khi bấm

def on_button_release(event):
    event.widget.config(background="#005c99", relief="flat", borderwidth=0)  # Khi nhả nút
    event.widget.config(foreground="#ffffff")  # Đổi lại màu chữ khi nhả

# Gán sự kiện cho nút
btn_select.bind("<Enter>", on_hover_in)
btn_select.bind("<Leave>", on_hover_out)
btn_select.bind("<ButtonPress-1>", on_button_press)
btn_select.bind("<ButtonRelease-1>", on_button_release)

btn_next.bind("<Enter>", on_hover_in)
btn_next.bind("<Leave>", on_hover_out)
btn_next.bind("<ButtonPress-1>", on_button_press)
btn_next.bind("<ButtonRelease-1>", on_button_release)

# Khởi chạy ứng dụng
root.mainloop()
