FROM archlinux:base-devel AS libriichi_build

# 使用国内镜像加速
RUN echo 'Server = https://mirrors.aliyun.com/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist && \
    echo 'Server = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch' >> /etc/pacman.d/mirrorlist && \
    pacman -Syu --noconfirm --needed rust python && \
    pacman -Scc

WORKDIR /
COPY Cargo.toml Cargo.lock .
COPY libriichi libriichi
COPY exe-wrapper exe-wrapper

RUN cargo build -p libriichi --lib --release

# ----- 最终运行阶段 -----
FROM archlinux:base

RUN echo 'Server = https://mirrors.aliyun.com/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist && \
    echo 'Server = https://mirrors.tuna.tsinghua.edu.cn/archlinux/$repo/os/$arch' >> /etc/pacman.d/mirrorlist && \
    pacman -Syu --noconfirm --needed python python-pip && \
    pacman -Scc

RUN python -m pip install --no-cache-dir --break-system-packages torch toml tqdm tensorboard -i https://mirrors.aliyun.com/pypi/simple/

WORKDIR /mortal
COPY mortal .
COPY --from=libriichi_build /target/release/libriichi.so .

# 使用内嵌文件方式创建准确的 config.toml
RUN cat > config.toml <<'EOF'
[control]
state_file = '/mnt/mortal.pth'

[resnet]
conv_channels = 192
num_blocks = 40
enable_bn = true
bn_momentum = 0.99
EOF

VOLUME /mnt

ENTRYPOINT ["python", "mortal.py"]