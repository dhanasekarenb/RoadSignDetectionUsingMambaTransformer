import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler  # ✅ Correct Import

torch.backends.cudnn.benchmark = True  # Optimized CUDA kernels
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")

# ✅ Optimized Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False, padding_mode='reflect')
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, out_channels)  
        self.act = nn.SiLU(inplace=True)  # In-place activation

        # He Initialization
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)  

# ✅ Optimized Transformer Block with Flash Attention (if available)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(approximate="tanh"),  
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(0.15)  # Prevent overfitting

    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.dropout(self.fc(self.norm2(x)))
        return x

# ✅ Optimized Model
class MambaTransformerClassifier(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.layer1 = DepthwiseSeparableConv(3, 64)
        self.layer2 = DepthwiseSeparableConv(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((8, 8))  # ✅ Adaptive pooling

        # Flatten size calculation
        flatten_size = 128 * 8 * 8
        self.embedding = nn.Linear(flatten_size, 128)
        self.transformer = TransformerBlock(128, 4)
        self.fc = nn.Linear(128, num_classes)

        # Weight Initialization
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1).contiguous()
        x = self.embedding(x)

        x = x.unsqueeze(1)  
        x = self.transformer(x)
        x = x.squeeze(1)

        return self.fc(x)

# ✅ Optimized Training Function with Mixed Precision & Error Handling
scaler = GradScaler(device="cuda")  # 🚀 Fixed FutureWarning
def train_step(model, dataloader, criterion, optimizer, device):
    model.train()
    for batch_idx, (images, labels) in enumerate(dataloader):
        try:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16):  # 🚀 Fixed FutureWarning
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if batch_idx % 10 == 0:  # Print every 10 batches to reduce overhead
                print(f"🎯 Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"🔥 CUDA OOM at Batch {batch_idx+1}! Clearing Cache...")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"❌ Error in Batch {batch_idx+1}: {e}")
            import traceback
            print(traceback.format_exc())
            exit(1)

# ✅ Optimized Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MambaTransformerClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)  

if torch.cuda.is_available():
    model = torch.compile(model)  # 🚀 Torch.compile for Speed Optimization
