#! /bin/bash
export PATH="$PATH":$(pwd)/bin
rm -rf screenlog.0
screen -d -m -L ./bin/prepare_context.sh
