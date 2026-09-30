#!/usr/bin/env bash
# =============================================================================
# Mac Wi-Fi switch with auto-rollback  (ASCII-only, macOS bash 3.2 safe)
#
# Usage:
#     bash wifi_switch.sh board      -> Xiaomi_A389  (192.168.31.123)
#     bash wifi_switch.sh platform   -> ChinaNet-DHEr (192.168.1.8)
#     bash wifi_switch.sh status
# =============================================================================

IFACE="en0"
BOARD_SSID="Xiaomi_A389"
BOARD_HOST="192.168.31.123"
PLATFORM_SSID="ChinaNet-DHEr"
PLATFORM_HOST="192.168.1.8"
TARGET="$1"

current_ssid() {
    system_profiler SPAirPortDataType 2>/dev/null \
        | awk '/Current Network Information:/{getline; gsub(/^ +|:$/,""); print; exit}'
}

probe() {
    ping -c 2 -t 3 "$1" >/dev/null 2>&1 && return 0
    nc -z -G 3 "$1" 22 >/dev/null 2>&1 && return 0
    return 1
}

do_switch() {
    SSID="$1"
    HOST="$2"
    NAME="$3"
    BEFORE="$(current_ssid)"

    echo "[INFO] current SSID: ${BEFORE:-unknown}"
    if [ "$BEFORE" = "$SSID" ]; then
        echo "[INFO] already on $SSID"
    else
        echo "[INFO] switching to $SSID ..."
        networksetup -setairportnetwork "$IFACE" "$SSID" >/dev/null 2>&1
        sleep 10
    fi

    echo "[INFO] verifying $NAME reachability ($HOST) ..."
    i=1
    while [ $i -le 6 ]; do
        if probe "$HOST"; then
            echo "[OK] $NAME reachable | SSID=$(current_ssid) IP=$(ipconfig getifaddr $IFACE 2>/dev/null)"
            return 0
        fi
        sleep 4
        i=$((i+1))
    done

    echo "[FAIL] $NAME not reachable"
    if [ -n "$BEFORE" ] && [ "$BEFORE" != "$SSID" ]; then
        echo "[ROLLBACK] back to $BEFORE ..."
        networksetup -setairportnetwork "$IFACE" "$BEFORE" >/dev/null 2>&1
        sleep 10
        echo "[ROLLBACK] SSID=$(current_ssid) IP=$(ipconfig getifaddr $IFACE 2>/dev/null)"
    fi
    return 1
}

case "$TARGET" in
    board)    do_switch "$BOARD_SSID" "$BOARD_HOST" "BOARD" ;;
    platform) do_switch "$PLATFORM_SSID" "$PLATFORM_HOST" "PLATFORM" ;;
    status)
        echo "SSID : $(current_ssid)"
        echo "IP   : $(ipconfig getifaddr $IFACE 2>/dev/null)"
        probe "$BOARD_HOST" && echo "BOARD $BOARD_HOST : reachable" || echo "BOARD $BOARD_HOST : unreachable"
        probe "$PLATFORM_HOST" && echo "PLATFORM $PLATFORM_HOST : reachable" || echo "PLATFORM $PLATFORM_HOST : unreachable"
        ;;
    *) echo "usage: bash $0 {board|platform|status}"; exit 2 ;;
esac
