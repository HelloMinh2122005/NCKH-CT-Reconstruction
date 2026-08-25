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

        # Chiếu Q, K, V và tách thành Multi-Head: (B, H, N, d)
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        # Chuẩn hóa QK Norm nếu được kích hoạt
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        # Tính toán phân đoạn và giãn nở (Segment & Dilation)
        segment_span = w * r
        if segment_span > 0 and (N % segment_span == 0):
            num_segments = N // segment_span
            
            # Tái cấu trúc tensor để gom các token cách nhau r bước vào cùng segment w
            q_dilated = (
                q.view(B, H, num_segments, w, r, d)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(B, H, num_segments * r, w, d)
            )
            k_dilated = (
                k.view(B, H, num_segments, w, r, d)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(B, H, num_segments * r, w, d)
            )
            v_dilated = (
                v.view(B, H, num_segments, w, r, d)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(B, H, num_segments * r, w, d)
            )

            # Tính Attention Weights theo từng segment cục bộ
            attn_scores = torch.matmul(q_dilated, k_dilated.transpose(-1, -2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out_dilated = torch.matmul(attn_weights, v_dilated)

            # Ghép nối các segment ngược trở lại chuỗi ban đầu
            out = (
                out_dilated.view(B, H, num_segments, r, w, d)
                .permute(0, 1, 2, 4, 3, 5)
                .reshape(B, H, N, d)
            )
        else:
            # Fallback sang standard multi-head attention nếu chiều dài chuỗi N không chia hết cho w*r
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # Chiếu qua lớp tuyến tính đầu ra
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
