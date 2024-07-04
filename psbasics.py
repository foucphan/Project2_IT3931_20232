import imageio.v2 as imageio
import sys
import numpy as np
from scipy import misc
from matplotlib import pyplot as plt


def sanitise_image(image):
    return (image / 255).flatten()


def ps_basic_ols(images, L, original_size):
    # Chuyển đổi hình ảnh thành các vector hàng
    images = list(map(sanitise_image, images))
    images = np.vstack(images)

    # Chuẩn hóa các vector ánh sáng
    L = L / np.linalg.norm(L, ord=2, axis=1, keepdims=True)

    # Giải phương trình G = N'*rho sử dụng phương pháp bình phương tối thiểu
    norm_sln = np.linalg.pinv(L.T.dot(L)).dot(L.T)

    # Tính toán G
    G = np.einsum("ij,il", norm_sln, images)

    # Tính toán albedo
    rho = np.linalg.norm(G, axis=0)

    # Tính toán bản đồ pháp tuyến
    N = np.divide(G, np.vstack([rho] * 3))

    # Đưa kết quả về dạng hình ảnh
    rho = rho.reshape(original_size)
    N = N.T.reshape(original_size[0], original_size[1], 3)

    return N, rho


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("USAGE: psbasics <lights_filename> <image_1_filename> <image_2_filename> <image_3_filename> ...")
        sys.exit()

    # Đọc các vector ánh sáng
    L = np.loadtxt(sys.argv[1]).T

    if len(sys.argv) - 2 != L.shape[1]:
        raise ValueError('Error: The number of light vectors does not match the number of input images.')

    # Đọc và xử lý các hình ảnh đầu vào
    images = []
    org_size = None
    for i in range(L.shape[1]):
        image_path = sys.argv[i + 2]
        image = imageio.imread(image_path)

        if image.ndim > 2:
            image = image.mean(axis=2)

        org_size = image.shape
        images.append(image.flatten())

    # Chạy thuật toán PS
    N, rho = ps_basic_ols(images, L, org_size)

    # Chuyển giá trị của N về phạm vi 0-255 và kiểu dữ liệu uint8
    N_display = ((N + 1) * 255 / 2).astype(np.uint8)

    # Lưu kết quả
    imageio.imsave("normal_map.png", N_display)
