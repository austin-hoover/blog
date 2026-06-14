#!/bin/bash

for dir in */; do
    dir="${dir%/}"  # strip trailing slash
    [[ "$dir" == _* ]] && continue
    new="${dir//_/-}"
    if [ "$dir" != "$new" ]; then
        mv -- "$dir" "$new" && echo "Renamed: $dir -> $new"
    fi
done
