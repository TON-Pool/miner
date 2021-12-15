#!/usr/bin/env bash

DIRNAME=$(dirname "$BASH_SOURCE")
. $DIRNAME/h-manifest.conf

temp=$(jq '.temp' <<< $gpu_stats)
fan=$(jq '.fan' <<< $gpu_stats)
[[ $cpu_indexes_array != '[]' ]] &&
    temp=$(jq -c "del(.$cpu_indexes_array)" <<< $temp) &&
    fan=$(jq -c "del(.$cpu_indexes_array)" <<< $fan)

STATS_FILE=/hive/miners/custom/${CUSTOM_NAME}-${CUSTOM_VERSION}/stats.json

if [[ ! -f $STATS_FILE ]]; then
    echo "stats.json not found"
else
    khs=$(jq .total $STATS_FILE)
    hs=$(jq .rates $STATS_FILE)
    uptime=$(jq .uptime $STATS_FILE)
    ac=$(jq .accepted $STATS_FILE)
    rj=$(jq .rejected $STATS_FILE)
    stats=$(jq -n \
        --argjson hs "$hs" --arg hs_units mhs \
        --argjson temp "$temp" --argjson fan "$fan" \
        --arg algo darkcoin \
        --arg uptime "$uptime" \
        --arg ver "$CUSTOM_VERSION" \
        --arg ac "$ac" --arg rj "$rj" \
        '{$hs, $hs_units, $temp, $fan, $algo, $uptime, $ver, ar: [$ac, $rj]}')
fi