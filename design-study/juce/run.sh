#!/bin/sh
set -eu
study_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cmake -S "$study_dir" -B "$study_dir/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$study_dir/build" --config Release -j 4
if [ "$(uname)" = Darwin ]; then
  open "$study_dir/build/PitcherStudy_artefacts/Release/Pitcher Design Study.app"
else
  "$study_dir/build/PitcherStudy_artefacts/Release/Pitcher Design Study"
fi
