#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl gnupg xauth util-linux unzip
install -m 0755 -d /etc/apt/keyrings
curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 120 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n' "$(dpkg --print-architecture)" "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list
curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 120 https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes --output /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 120 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's|deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://|g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
