"""
================================================================================
SCRIPT VẼ KIẾN TRÚC MẠNG: FISTA-Net / LEARN_MVA cho Limited-Angle CT (LA-CT)
Thư viện: PlotNeuralNet (TikZ LaTeX Generator)
Tham chiếu kiến trúc: docs/LEARN_MVA_diagram.html
Tác giả: AI Research Engineer - Dự án NCKH CT Reconstruction (MinhPD)
================================================================================
Mô tả luồng kiến trúc:
1. Giai đoạn khởi tạo (Input & Initialization):
   - Sinogram góc giới hạn: y in R^(1 x 64 x 512) (LA-120°, 64 views)
   - Tái tạo sơ bộ giải tích: FBP Ram-Lak -> x^(0) in R^(1 x 256 x 256)

2. Vòng lặp mở cuộn unrolling bậc 1 (Unrolled Stage k -> k+1):
   - Trạng thái hiện tại: x^(k) in R^(1 x 256 x 256)
   - Nhánh 1 (Vật lý - Data Consistency Gradient):
     + Chiếu thuận RayTransform: A(x^(k)) in R^(1 x 64 x 512)
     + Sai lệch Sinogram (Residual): A(x^(k)) - y
     + Chiếu ngược liên hợp (Adjoint): A^T(A x^(k) - y) in R^(1 x 256 x 256)
     + Bước nhảy vật lý: g_phys^(k) = alpha^(k) * A^T(A x^(k) - y)
   - Nhánh 2 (Tiên nghiệm học sâu - Sandwich Prior with Sequence Engine):
     + Tầng Conv1 + ReLU: 1 -> 48 channels, kernel 5x5 -> R^(48 x 256 x 256)
     + Token hóa Patch T_(2x2): patch 2x2 -> 128x128 patches, N = 16,384 tokens, d = 192
     + Sequence Engine (MVA / Longformer / LongNet / Mamba): Attention chuỗi dài O(N)
     + Khôi phục không gian 2D T_(2x2)^(-1): -> R^(48 x 256 x 256)
     + Tầng Conv2 + ReLU: 48 -> 48 channels, kernel 5x5
     + Tầng Conv3 (Linear Output): 48 -> 1 channel -> Gradient tiên nghiệm nabla R_theta^(k)
   - Tổng hợp Gradient & Cập nhật nghiệm:
     + Nút cộng Gradient (+): g_total = g_phys + nabla R_theta
     + Cập nhật nghiệm x^(k+1) = x^(k) - g_total kèm kết nối tắt (Skip Connection) từ x^(k)

3. Mở cuộn đa tầng & Tái tạo cuối (Cascade & Supervised Loss):
   - Chuỗi mở cuộn 14 stages (Unrolling x14)
   - Ảnh tái tạo cuối cùng x^(14) in R^(1 x 256 x 256)
   - Hàm mất mát giám sát End-to-End MSE Loss: L_MSE vs Ground-Truth x*
================================================================================
"""

import sys
import os

# Thêm đường dẫn thư mục gốc PlotNeuralNet để nạp module pycore
sys.path.append('../')
from pycore.tikzeng import *

# Định nghĩa danh sách các thành phần kiến trúc mạng theo chuẩn TikZ của PlotNeuralNet
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # ==========================================================================
    # 1. GIAI ĐOẠN ĐẦU VÀO & KHỞI TẠO (INPUT & ANALYTICAL RECONSTRUCTION)
    # ==========================================================================
    # 1.1. Sinogram đo đạc thực tế góc giới hạn y: [B, 1, 64, 512] (LA-120°, 64 views)
    to_Conv(
        name="sino_in",
        s_filer="512",
        n_filer="64",
        offset="(0,0,0)",
        to="(0,0,0)",
        width=2.5,
        height=14,
        depth=36,
        caption=r"Sinogram $y$ (LA-$120^\circ$)"
    ),

    # 1.2. Khởi tạo sơ bộ bằng FBP lọc Ram-Lak: x^(0) = A_FBP(y) in [B, 1, 256, 256]
    to_Conv(
        name="fbp_init",
        s_filer="256",
        n_filer="1",
        offset="(2.5,0,0)",
        to="(sino_in-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"FBP Init $x^{(0)}$"
    ),
    to_connection("sino_in", "fbp_init"),

    # ==========================================================================
    # 2. VÒNG LẶP MỞ CUỘN UNROLLING (UNROLLED STAGE k -> k+1)
    # ==========================================================================
    # 2.1. Trạng thái ảnh hiện tại x^(k): [B, 1, 256, 256]
    to_Conv(
        name="x_k",
        s_filer="256",
        n_filer="1",
        offset="(2.5,0,0)",
        to="(fbp_init-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"State $x^{(k)}$"
    ),
    to_connection("fbp_init", "x_k"),

    # --------------------------------------------------------------------------
    # NHÁNH VẬT LÝ A: DATA CONSISTENCY (CHIẾU THUẬN & LIÊN HỢP NGƯỢC RADON)
    # Đặt lệch lên phía trên (dy = +4.5)
    # --------------------------------------------------------------------------
    # 2.2. Chiếu thuận RayTransform: A(x^(k)) in [B, 1, 64, 512]
    to_Conv(
        name="radon_fwd",
        s_filer="512",
        n_filer="64",
        offset="(3.5, 4.5, 0)",
        to="(x_k-east)",
        width=2.5,
        height=14,
        depth=36,
        caption=r"Forward $A x^{(k)}$"
    ),
    to_connection("x_k", "radon_fwd"),

    # 2.3. Sai lệch Sinogram (Residual): A(x^(k)) - y in [B, 1, 64, 512]
    to_Conv(
        name="sino_res",
        s_filer="512",
        n_filer="64",
        offset="(3.2, 0, 0)",
        to="(radon_fwd-east)",
        width=2.5,
        height=14,
        depth=36,
        caption=r"Resid $A x - y$"
    ),
    to_connection("radon_fwd", "sino_res"),

    # 2.4. Chiếu ngược liên hợp chuẩn: A^T(A x - y) in [B, 1, 256, 256]
    to_Conv(
        name="radon_adj",
        s_filer="256",
        n_filer="1",
        offset="(3.2, 0, 0)",
        to="(sino_res-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"Adjoint $A^T(\cdot)$"
    ),
    to_connection("sino_res", "radon_adj"),

    # 2.5. Gradient vật lý hoàn chỉnh có bước nhảy: g_phys^(k) = alpha^(k) * A^T(A x - y)
    to_Conv(
        name="grad_phys",
        s_filer="256",
        n_filer="1",
        offset="(3.2, 0, 0)",
        to="(radon_adj-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"Phys-Grad $g_{\mathrm{phys}}^{(k)}$"
    ),
    to_connection("radon_adj", "grad_phys"),

    # --------------------------------------------------------------------------
    # NHÁNH TIÊN NGHIỆM HỌC SÂU B: SANDWICH PRIOR VỚI SEQUENCE ENGINE (MVA)
    # Đặt lệch xuống phía dưới (dy = -4.5)
    # --------------------------------------------------------------------------
    # 2.6. Tầng trích xuất đặc trưng Conv1 + ReLU: 1 -> 48 kênh, 256x256
    to_ConvConvRelu(
        name="conv1",
        s_filer=256,
        n_filer=(1, 48),
        offset="(3.5, -4.5, 0)",
        to="(x_k-east)",
        width=(1.6, 3.8),
        height=36,
        depth=36,
        caption=r"Conv1 + ReLU"
    ),
    to_connection("x_k", "conv1"),

    # 2.7. Token hóa Patch không trùng lặp T_(2x2): N = 16,384 tokens, d = 48*4 = 192 dims
    to_Conv(
        name="tokenize",
        s_filer="128",
        n_filer="192",
        offset="(1.8, 0, 0)",
        to="(conv1-east)",
        width=5.5,
        height=24,
        depth=24,
        caption=r"Tokenize $T_{2\times2}$"
    ),
    to_connection("conv1", "tokenize"),

    # 2.8. Cỗ máy xử lý chuỗi dài Sequence Engine (Multi-View Attention / Longformer / LongNet / Mamba)
    to_SoftMax(
        name="seq_engine",
        s_filer="16384",
        offset="(1.8, 0, 0)",
        to="(tokenize-east)",
        width=4.0,
        height=20,
        depth=20,
        opacity=0.85,
        caption=r"MVA Engine $\mathcal{O}(N)$"
    ),
    to_connection("tokenize", "seq_engine"),

    # 2.9. Khôi phục kích thước không gian Untokenize T_(2x2)^(-1): gấp lại [B, 48, 256, 256]
    to_Conv(
        name="untokenize",
        s_filer="256",
        n_filer="48",
        offset="(1.8, 0, 0)",
        to="(seq_engine-east)",
        width=3.8,
        height=36,
        depth=36,
        caption=r"Untokenize $T_{2\times2}^{-1}$"
    ),
    to_connection("seq_engine", "untokenize"),

    # 2.10. Tầng lọc ảnh nâng cao Conv2 + ReLU: 48 -> 48 kênh, 256x256
    to_ConvConvRelu(
        name="conv2",
        s_filer=256,
        n_filer=(48, 48),
        offset="(1.8, 0, 0)",
        to="(untokenize-east)",
        width=(3.8, 3.8),
        height=36,
        depth=36,
        caption=r"Conv2 + ReLU"
    ),
    to_connection("untokenize", "conv2"),

    # 2.11. Tầng chiếu đầu ra Conv3: 48 -> 1 kênh, tạo gradient tiên nghiệm nabla R_theta^(k)
    to_Conv(
        name="conv3",
        s_filer="256",
        n_filer="1",
        offset="(1.8, 0, 0)",
        to="(conv2-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"Prior $\nabla R_\theta^{(k)}$"
    ),
    to_connection("conv2", "conv3"),

    # --------------------------------------------------------------------------
    # TỔNG HỢP GRADIENT & BƯỚC CẬP NHẬT NGHIỆM
    # --------------------------------------------------------------------------
    # 2.12. Nút cộng Gradient (+): Tổng hợp gradient vật lý và gradient tiên nghiệm học sâu
    to_Sum(
        name="sum_grad",
        offset="(2.5, 4.5, 0)",
        to="(conv3-east)",
        radius=2.5,
        opacity=0.75
    ),
    to_connection("grad_phys", "sum_grad"),
    to_connection("conv3", "sum_grad"),

    # 2.13. Cập nhật trạng thái stage tiếp theo: x^(k+1) = x^(k) - [g_phys + nabla R_theta]
    to_Conv(
        name="x_next",
        s_filer="256",
        n_filer="1",
        offset="(2.6, 0, 0)",
        to="(sum_grad-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"State $x^{(k+1)}$"
    ),
    to_connection("sum_grad", "x_next"),

    # 2.14. Đường kết nối tắt Residual / Momentum từ x^(k) tới x^{(k+1)} vượt qua phía trên
    to_skip(
        of="x_k",
        to="x_next",
        pos=1.6
    ),

    # ==========================================================================
    # 3. CHUỖI MỞ CUỘN ĐA TẦNG & TÁI TẠO CUỐI CÙNG (CASCADE & SUPERVISED LOSS)
    # ==========================================================================
    # 3.1. Khối lặp mở cuộn sâu 14 stages cascading (k = 0, 1, ..., 13)
    to_Conv(
        name="unroll_cascade",
        s_filer="256",
        n_filer="14",
        offset="(2.6, 0, 0)",
        to="(x_next-east)",
        width=4.5,
        height=36,
        depth=36,
        caption=r"Unrolling $\times 14$"
    ),
    to_connection("x_next", "unroll_cascade"),

    # 3.2. Ảnh CT tái tạo cuối cùng x^(14): [B, 1, 256, 256] sạch sọc nhiễu
    to_Conv(
        name="x_final",
        s_filer="256",
        n_filer="1",
        offset="(2.6, 0, 0)",
        to="(unroll_cascade-east)",
        width=1.6,
        height=36,
        depth=36,
        caption=r"Output $x^{(14)}$"
    ),
    to_connection("unroll_cascade", "x_final"),

    # 3.3. Hàm mất mát giám sát đầu cuối End-to-End Supervised Loss (MSE vs Ground Truth x*)
    to_Conv(
        name="loss_mse",
        s_filer="1",
        n_filer="1",
        offset="(2.2, 0, 0)",
        to="(x_final-east)",
        width=1.5,
        height=18,
        depth=18,
        caption=r"$\mathcal{L}_{\mathrm{MSE}}$"
    ),
    to_connection("x_final", "loss_mse"),

    to_end()
]


def main():
    """Hàm thực thi chính: tạo file mã nguồn LaTeX TikZ tương ứng."""
    namefile = str(sys.argv[0]).split('.')[0]
    output_tex = namefile + '.tex'
    to_generate(arch, output_tex)
    print(f"🎉 [PlotNeuralNet] Đã sinh thành công file LaTeX TikZ: {output_tex}")


if __name__ == '__main__':
    main()
