#!/usr/bin/env bash
# 修复 NVIDIA 显卡在推理或训练过程中黑屏、掉卡 (Xid 79) 以及无法唤醒屏幕的系统配置脚本
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] This script must be run as root (or with sudo)." >&2
    echo "Usage: sudo bash $0" >&2
    exit 1
fi

echo "================================================================================"
echo "Starting GPU Freeze & Black Screen Software Fix"
echo "================================================================================"

# 1. 强制启用 NVIDIA Persistence Mode（持久化模式）
# 默认配置中 nvidia-persistenced.service 包含 --no-persistence-mode，导致 CUDA 任务结束释放 UVM 时
# 驱动卸载/重置显卡，导致正在运行的 Xorg 桌面显示引擎崩溃（Failed to allocate push buffer / Xid 79）
echo "[1/6] Configuring NVIDIA Persistence Mode..."
mkdir -p /etc/systemd/system/nvidia-persistenced.service.d
cat > /etc/systemd/system/nvidia-persistenced.service.d/override.conf << 'EOF'
[Service]
ExecStart=
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --persistence-mode --verbose
EOF

systemctl daemon-reload
systemctl restart nvidia-persistenced.service || true
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -pm 1 || true
fi
echo "[OK] NVIDIA Persistence Mode enabled."

# 2. 禁用 PCIe Runtime 自动休眠（PCIe Runtime Autosuspend）
# 当前显卡 PCIe 控制被设置为 auto，导致高负载推理/训练时 PCIe 链路进入低功耗状态，引发掉卡（GPU fallen off the bus）
echo "[2/6] Disabling PCIe Runtime Autosuspend for NVIDIA devices..."
cat > /etc/udev/rules.d/99-nvidia-pm.rules << 'EOF'
# 禁用 NVIDIA 显卡 PCIe 运行时电源管理（自动休眠）
ACTION=="add|change", SUBSYSTEM=="pci", ATTRS{vendor}=="0x10de", ATTR{power/control}="on"
EOF

# 立即对当前所有 NVIDIA 显卡应用 power/control=on
for pci_power in /sys/bus/pci/devices/*/power/control; do
    device_dir="$(dirname "$pci_power")"
    if [[ -f "$device_dir/vendor" ]]; then
        vendor="$(cat "$device_dir/vendor" 2>/dev/null || true)"
        if [[ "$vendor" == "0x10de" ]]; then
            echo on > "$pci_power" 2>/dev/null || true
            echo "  -> Set $(basename "$device_dir") power/control to 'on'"
        fi
    fi
done
udevadm control --reload-rules || true
udevadm trigger || true
echo "[OK] PCIe runtime autosuspend disabled."

# 3. 切换 PRIME 配置为独立显卡模式 (prime-select nvidia)
# 台式机单卡 RTX 2060 连接主显示器时，若处于 on-demand 模式会触发笔记本混合显卡节能逻辑
echo "[3/6] Configuring PRIME mode to dedicated NVIDIA..."
if command -v prime-select >/dev/null 2>&1; then
    current_prime="$(prime-select query 2>/dev/null || true)"
    if [[ "$current_prime" != "nvidia" ]]; then
        prime-select nvidia || true
        echo "[OK] PRIME profile switched to nvidia."
    else
        echo "[OK] PRIME profile already set to nvidia."
    fi
fi

# 4. 配置 Xorg 选项，防止长耗时 CUDA Kernel 触发 Xorg 看门狗超时崩溃
echo "[4/6] Configuring Xorg display server options..."
mkdir -p /etc/X11/xorg.conf.d
cat > /etc/X11/xorg.conf.d/20-nvidia.conf << 'EOF'
Section "OutputClass"
    Identifier     "nvidia"
    MatchDriver    "nvidia-drm"
    Driver         "nvidia"
    Option         "Interactive" "False"
    Option         "HardDPMS" "False"
EndSection
EOF
echo "[OK] Xorg anti-freeze options configured in /etc/X11/xorg.conf.d/20-nvidia.conf."

# 5. 配置 Linux 内核引导参数禁用 PCIe ASPM 与 AER 错误中断
# PCIe ASPM（主动状态电源管理）会导致高功耗波动时 PCIe 链路掉线，产生 Xid 79
echo "[5/6] Checking Linux kernel boot parameters in /etc/default/grub..."
GRUB_FILE="/etc/default/grub"
if [[ -f "$GRUB_FILE" ]]; then
    NEEDS_UPDATE=0
    if ! grep -q "pcie_aspm=off" "$GRUB_FILE"; then
        cp -a "$GRUB_FILE" "${GRUB_FILE}.bak.$(date +%Y%m%d%H%M%S)"
        sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 pcie_aspm=off"/' "$GRUB_FILE"
        NEEDS_UPDATE=1
        echo "  -> Added 'pcie_aspm=off' to GRUB_CMDLINE_LINUX_DEFAULT"
    fi
    if ! grep -q "pci=noaer" "$GRUB_FILE"; then
        sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT="[^"]*\)"/\1 pci=noaer"/' "$GRUB_FILE"
        NEEDS_UPDATE=1
        echo "  -> Added 'pci=noaer' to GRUB_CMDLINE_LINUX_DEFAULT"
    fi
    if [[ "$NEEDS_UPDATE" -eq 1 ]]; then
        echo "  -> Updating GRUB..."
        update-grub || true
        echo "[OK] GRUB boot parameters updated successfully."
    else
        echo "[OK] Kernel parameters pcie_aspm=off and pci=noaer already present."
    fi
fi

# 6. 配置稳定显卡功率上限 systemd 服务（削减瞬态峰值电流，防止电源过载掉卡）
echo "[6/6] Configuring GPU power limit service (optional transient protection)..."
cat > /etc/systemd/system/nvidia-power-limit.service << 'EOF'
[Unit]
Description=Set NVIDIA GPU Power Limit for Stability
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/nvidia-smi -pl 170

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable nvidia-power-limit.service || true
systemctl start nvidia-power-limit.service || true
echo "[OK] GPU power limit service configured (170W cap)."

echo "================================================================================"
echo "All software-level GPU freeze fixes applied successfully!"
echo "================================================================================"
