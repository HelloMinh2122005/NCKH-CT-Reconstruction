"""
================================================================================
MODULE DILATED ATTENTION (LONGNET) CHO KIẾN TRÚC SOLAR_LongNet
Dự án: Nghiên cứu Tái tạo Ảnh Cắt lớp CT Góc Giới hạn (Limited-Angle CT Reconstruction)
Tác giả: MinhPD — Nhóm Nghiên cứu Tái tạo Ảnh Y tế (VNU-HCM UIT)
Mô tả:
    - Module Dilated Attention từ kiến trúc LongNet (Microsoft Research, 2023).
    - Mở rộng tầm quan sát không gian O(N) thông qua cơ chế phân đoạn và lấy mẫu giãn nở.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedAttention(nn.Module):
    """
    Module Dilated Attention từ kiến trúc LongNet (Microsoft Research, 2023).
    
    Nguyên lý hoạt động:
    - Cho phép Transformer mở rộng khả năng tiếp nhận chuỗi token cực dài O(N) thông qua cơ chế 
      phân đoạn (segmentation) và lấy mẫu giãn nở (dilation rate r).
    - Thay vì tính ma trận tương quan đầy đủ N x N (độ phức tạp O(N^2)), Dilated Attention gom nhóm
      các token cách nhau r bước vào cùng một đoạn segment kích thước w.
    - Nhờ đó, mô hình vừa quan sát được ngữ cảnh toàn cục tầm xa, vừa giữ cho chi phí tính toán và bộ nhớ
      ở mức tuyến tính O(N).
      
    Tham số khởi tạo:
    - dim (int): Kích thước vector nhúng của token (mặc định = 192).
    - heads (int): Số attention heads (mặc định = 4).
    - dilation_rate (int): Tỷ lệ giãn nở khoảng cách giữa các token (mặc định = 1).
    - segment_size (int): Kích thước của mỗi phân đoạn segment w (mặc định = 16).
    - dropout (float): Tỷ lệ dropout áp dụng lên ma trận attention weights (mặc định = 0.1).
    - qk_norm (bool): Có chuẩn hóa L2 cho vector Query và Key trước khi nhân vô hướng hay không (mặc định = True).
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dilation_rate: int = 1,
        segment_size: int = 16,
        dropout: float = 0.1,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.dilation_rate = dilation_rate
        self.segment_size = segment_size
        self.scale = self.head_dim ** -0.5
        self.qk_norm = qk_norm

        # Các lớp biến đổi tuyến tính chiếu Query, Key, Value
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Luồng tính toán Forward của Dilated Attention:
        Đầu vào: x có shape (Batch_size, Sequence_length N, Embedding_dim D)
        Đầu ra: Tensor có cùng shape (B, N, D)
        """
        B, N, D = x.shape
        H = self.heads
        d = self.head_dim
        r = self.dilation_rate
        w = self.segment_size

        # Bước 1: Chiếu tuyến tính Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Bước 2: Chuẩn hóa L2 nếu bật qk_norm
        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        # Phân rã chiều đặc trưng cho multi-head attention: (B, N, H, d) -> (B, H, N, d)
        q = q.view(B, N, H, d).transpose(1, 2)
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # Bước 3: Thu thập các token theo tỷ lệ giãn nở r
        if r > 1:
            q = q.view(B, H, -1, r, d).transpose(2, 3).reshape(B, H * r, -1, d)
            k = k.view(B, H, -1, r, d).transpose(2, 3).reshape(B, H * r, -1, d)
            v = v.view(B, H, -1, r, d).transpose(2, 3).reshape(B, H * r, -1, d)
            curr_H = H * r
        else:
            curr_H = H

        # Bước 4: Phân đoạn chuỗi dài thành các segment kích thước w
        seq_len = q.shape[2]
        num_segments = seq_len // w

        if num_segments > 0:
            q_seg = q.view(B, curr_H, num_segments, w, d)
            k_seg = k.view(B, curr_H, num_segments, w, d)
            v_seg = v.view(B, curr_H, num_segments, w, d)

            # Tính Attention Map nội bộ trong từng phân đoạn segment
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out_seg = torch.matmul(attn_weights, v_seg)

            out = out_seg.view(B, curr_H, seq_len, d)
        else:
            # Fallback nếu chuỗi ngắn hơn kích thước segment w
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # Bước 5: Khôi phục cấu trúc chuỗi ban đầu nếu r > 1
        if r > 1:
            out = (
                out.view(B, H, r, -1, d)
                .transpose(2, 3)
                .reshape(B, H, seq_len * r, d)
            )

        # Gộp các Head và chiếu qua out_proj
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return out
