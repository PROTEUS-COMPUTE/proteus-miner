#!/usr/bin/env bash
# Report per-card stats to the Hive agent.
#
# The agent sources this and expects two variables:
#   $khs    a single number, the rig total
#   $stats  a JSON object, one entry per card
#
# PROTEUS does not hash. The number that means anything here is generation
# throughput in tokens per second, so that is what goes in "hs". Units are
# declared as "hs" because the agent only accepts its own scale names; the web UI
# will label it H/s and the value is tokens/s.
#
# Cards are matched to the rig by PCI bus number, so hs, temp, fan and
# bus_numbers are all built in the same order and always the same length.
. /hive/miners/custom/proteus/h-manifest.conf 2>/dev/null

khs=0
stats=""

# Which cards h-run.sh actually brought up, not which cards exist.
if [[ -s /run/proteus-gpus ]]; then
  mapfile -t IDS < /run/proteus-gpus
else
  mapfile -t IDS < <(docker ps --format '{{.Names}}' 2>/dev/null \
    | grep -oP '^proteus-vllm-\K[0-9]+' | sort -n)
fi

[[ "${#IDS[@]}" -eq 0 ]] && { echo "null"; return 0 2>/dev/null || exit 0; }

# index -> bus number, once, rather than per card.
declare -A BUS TEMP FAN
while IFS=, read -r i bus temp fan; do
  i="${i// /}"; bus="${bus// /}"
  # pci.bus_id looks like 00000000:0A:00.0, the agent wants the bus byte as an int.
  BUS[$i]=$((16#$(echo "$bus" | cut -d: -f2)))
  TEMP[$i]="${temp// /}"
  FAN[$i]="${fan// /}"
done < <(nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,fan.speed \
         --format=csv,noheader,nounits 2>/dev/null)

hs_a=(); temp_a=(); fan_a=(); bus_a=()
total=0
now=$(date +%s)
uptime=0

for i in "${IDS[@]}"; do
  [[ -z "$i" ]] && continue
  c="proteus-vllm-$i"

  # vLLM logs its throughput every few seconds. Reading the tail is cheaper and
  # more reliable than reaching into the container for /metrics, and it degrades
  # to 0 rather than erroring when the model is still loading.
  tps=$(docker logs --tail 40 "$c" 2>&1 \
        | grep -oP 'Avg generation throughput:\s*\K[0-9.]+' | tail -1)
  [[ -z "$tps" ]] && tps=0

  hs_a+=("$tps")
  temp_a+=("${TEMP[$i]:-0}")
  fan_a+=("${FAN[$i]:-0}")
  bus_a+=("${BUS[$i]:-0}")
  total=$(echo "$total $tps" | awk '{printf "%.1f", $1 + $2}')

  # Rig uptime is the oldest stack still running.
  s=$(docker inspect -f '{{.State.StartedAt}}' "$c" 2>/dev/null)
  if [[ -n "$s" ]]; then
    st=$(date -d "$s" +%s 2>/dev/null)
    [[ -n "$st" ]] && u=$((now - st)) && [[ "$u" -gt "$uptime" ]] && uptime="$u"
  fi
done

# How many cards are reachable from outside. A rig that serves inference nobody
# can reach earns nothing, so it belongs in the stats, not only in the logs.
reachable=0
for i in "${IDS[@]}"; do
  docker exec "proteus-miner-$i" python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(3)
sys.exit(0 if s.connect_ex(('127.0.0.1', 8091)) == 0 else 1)" >/dev/null 2>&1 \
    && reachable=$((reachable + 1))
done

khs=$(echo "$total" | awk '{printf "%.3f", $1 / 1000}')

# Assembled by hand rather than with jq. Every value here is a number we produced
# ourselves, so there is nothing to escape, and the agent's view of the rig should
# not hinge on a helper being installed.
join() { local IFS=,; echo "$*"; }

stats=$(cat <<EOF
{"hs":[$(join "${hs_a[@]}")],"hs_units":"hs","temp":[$(join "${temp_a[@]}")],"fan":[$(join "${fan_a[@]}")],"bus_numbers":[$(join "${bus_a[@]}")],"uptime":$uptime,"ver":"$CUSTOM_VERSION","algo":"inference","ar":[$reachable,$(( ${#IDS[@]} - reachable ))]}
EOF
)

# A malformed object makes the agent read the rig as offline, which is worse than
# reporting nothing at all. Only publish it if it parses.
if ! echo "$stats" | python3 -c 'import json,sys; json.load(sys.stdin)' 2>/dev/null; then
  khs=0
  stats="null"
fi
