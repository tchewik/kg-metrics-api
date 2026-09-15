#!/bin/sh
set -eu

# The published image is intentionally self-contained. Verify that a user did
# not mask baked model paths with empty volumes, then start without networking.
/usr/local/bin/verify-bundled-models

exec "$@"
