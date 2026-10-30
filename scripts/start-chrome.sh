#!/bin/sh
# Separate, signed-out Chrome for agent experiments. Never sign in to anything here.
exec "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --user-data-dir="$HOME/.jev-chrome" --remote-debugging-port=9333 \
  --no-first-run --no-default-browser-check about:blank
