#!/usr/bin/env bash
# 永久彻底禁用系统所有层面的黑屏、休眠、锁屏、DPMS节能与显卡掉线保护脚本
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "[ERROR] This script must be run as root (or with sudo)." >&2
    echo "Usage: sudo bash $0" >&2
    exit 1
fi

echo "================================================================================"
echo "Starting Comprehensive System-Wide Black Screen Permanent Elimination"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. Systemd 操作系统层：彻底屏蔽休眠、挂起、睡眠目标
# ------------------------------------------------------------------------------
echo "[1/6] Disabling Systemd Sleep, Suspend, and Hibernate targets..."
mkdir -p /etc/systemd/sleep.conf.d /etc/systemd/logind.conf.d

cat > /etc/systemd/sleep.conf.d/no-suspend.conf << 'EOF'
[Sleep]
AllowSuspend=no
AllowHibernation=no
AllowSuspendThenHibernate=no
AllowHybridSleep=no
EOF

cat > /etc/systemd/logind.conf.d/no-suspend.conf << 'EOF'
[Login]
IdleAction=ignore
IdleActionSec=0
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
EOF

systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null 2>&1 || true
echo "[OK] Systemd sleep/suspend targets masked and disabled."

# ------------------------------------------------------------------------------
# 2. X11 显示服务器层：禁止 Xorg 发送 DPMS 熄屏与黑屏屏保信号
# ------------------------------------------------------------------------------
echo "[2/6] Configuring X11 ServerFlags to permanently disable DPMS & blanking..."
mkdir -p /etc/X11/xorg.conf.d /etc/X11/Xsession.d

cat > /etc/X11/xorg.conf.d/10-no-dpms.conf << 'EOF'
Section "ServerFlags"
    Option "BlankTime" "0"
    Option "StandbyTime" "0"
    Option "SuspendTime" "0"
    Option "OffTime" "0"
    Option "DPMS" "false"
EndSection
EOF

cat > /etc/X11/xorg.conf.d/20-nvidia.conf << 'EOF'
Section "OutputClass"
    Identifier     "nvidia"
    MatchDriver    "nvidia-drm"
    Driver         "nvidia"
    Option         "Interactive" "False"
    Option         "HardDPMS" "False"
EndSection
EOF

# 全局 Xsession 登录挂载脚本：无论何种用户或桌面管理器启动 X 会话，均强制关闭 DPMS
cat > /etc/X11/Xsession.d/99no-dpms << 'EOF'
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
EOF
chmod +x /etc/X11/Xsession.d/99no-dpms

echo "[OK] Xorg server-level DPMS and screensaver blanking permanently disabled."

# ------------------------------------------------------------------------------
# 3. 桌面环境与登录管理器层（GNOME 全局 DConf + GDM3 登录界面）
# ------------------------------------------------------------------------------
echo "[3/6] Setting DConf global defaults (GNOME users & GDM3 login screen)..."
mkdir -p /etc/dconf/profile /etc/dconf/db/local.d /etc/dconf/db/gdm.d

# 配置系统用户 DConf Profile
cat > /etc/dconf/profile/user << 'EOF'
user-db:user
system-db:local
EOF

cat > /etc/dconf/db/local.d/00-no-black-screen << 'EOF'
[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/desktop/screensaver]
lock-enabled=false
ubuntu-lock-on-suspend=false
idle-activation-enabled=false

[org/gnome/settings-daemon/plugins/power]
idle-dim=false
sleep-inactive-ac-type='nothing'
sleep-inactive-ac-timeout=0
sleep-inactive-battery-type='nothing'
sleep-inactive-battery-timeout=0
power-button-action='nothing'
EOF

# 配置 GDM3 欢迎界面 DConf Profile（防止未登录或锁屏界面黑屏）
cat > /etc/dconf/profile/gdm << 'EOF'
user-db:user
system-db:gdm
file-db:/usr/share/gdm/greeter-dconf-defaults
EOF

cat > /etc/dconf/db/gdm.d/00-no-black-screen << 'EOF'
[org/gnome/desktop/session]
idle-delay=uint32 0

[org/gnome/settings-daemon/plugins/power]
sleep-inactive-ac-type='nothing'
sleep-inactive-ac-timeout=0
EOF

if command -v dconf >/dev/null 2>&1; then
    dconf update || true
fi
echo "[OK] DConf global settings applied for all users and GDM3 greeter."

# ------------------------------------------------------------------------------
# 4. Linux 内核层：禁用终端黑屏 (consoleblank=0) 与 PCIe ASPM/AER
# ------------------------------------------------------------------------------
echo "[4/6] Configuring Linux kernel boot parameters (/etc/default/grub)..."
GRUB_FILE="/etc/default/grub"
if [[ -f "$GRUB_FILE" ]]; then
    NEEDS_UPDATE=0
    for param in "consoleblank=0" "pcie_aspm=off" "pci=noaer"; do
        if ! grep -q "$param" "$GRUB_FILE"; then
            sed -i "s/\\(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*\\)\"/\\1 ${param}\"/" "$GRUB_FILE"
            NEEDS_UPDATE=1
            echo "  -> Added '${param}' to GRUB_CMDLINE_LINUX_DEFAULT"
        fi
    done
    if [[ "$NEEDS_UPDATE" -eq 1 ]]; then
        echo "  -> Updating GRUB bootloader..."
        update-grub || true
        echo "[OK] GRUB updated with consoleblank=0, pcie_aspm=off, and pci=noaer."
    else
        echo "[OK] Kernel boot parameters already up to date."
    fi
fi

# ------------------------------------------------------------------------------
# 5. 显卡硬件与驱动层：启用 Persistence Mode，禁用 PCIe Autosuspend
# ------------------------------------------------------------------------------
echo "[5/6] Configuring NVIDIA hardware & driver persistence settings..."
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

cat > /etc/udev/rules.d/99-nvidia-pm.rules << 'EOF'
# 禁用 NVIDIA 显卡 PCIe 运行时电源管理（自动休眠）
ACTION=="add|change", SUBSYSTEM=="pci", ATTRS{vendor}=="0x10de", ATTR{power/control}="on"
EOF

# 立即激活当前在线 PCI 显卡控制为 on
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

if command -v prime-select >/dev/null 2>&1; then
    prime-select nvidia >/dev/null 2>&1 || true
fi

# 可选：170W 功率上限服务平滑瞬态尖峰
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
echo "[OK] NVIDIA persistence and PCIe power lock enabled."

# ------------------------------------------------------------------------------
# 6. 虚拟终端 (TTY) 运行时防熄屏控制
# ------------------------------------------------------------------------------
echo "[6/6] Disabling TTY console blanking at runtime..."
cat > /etc/systemd/system/disable-console-blanking.service << 'EOF'
[Unit]
Description=Disable Console Blanking on TTY
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'TERM=linux setterm -blank 0 -powersave off -powerdown 0 </dev/tty1 >/dev/tty1 2>&1 || true'

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable disable-console-blanking.service || true
systemctl start disable-console-blanking.service || true
echo "[OK] Console blanking service enabled."

echo "================================================================================"
echo "Successfully configured system to PERMANENTLY NEVER go black screen!"
echo "Please reboot the machine once ('sudo reboot') to apply kernel & GRUB changes."
echo "================================================================================"
