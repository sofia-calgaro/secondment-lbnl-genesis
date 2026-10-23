#!/usr/bin/env bash
# Set up the locally built Remage Momentum environment on Perlmutter.

set -e

PROJECT="/global/cfs/cdirs/m2676/users/calgaro/uzh/secondment/optmap_demo"
INSTALL="$PROJECT/remage-momentum/install"

# Check that the local installation exists.
if [[ ! -x "$INSTALL/bin/remage" || ! -x "$INSTALL/bin/remage-cpp" ]]; then
    echo "ERROR: Local Remage installation not found at $INSTALL" >&2
    return 1 2>/dev/null || exit 1
fi

# Set library paths for Remage and Geant4.
export LD_LIBRARY_PATH="$INSTALL/lib:$PROJECT/.pixi/envs/default/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Prefer the local Remage executables.
export PATH="$INSTALL/bin:$PATH"

# Tell the Remage Python wrapper which C++ executable to use.
export REMAGE_CPP_PATH="$INSTALL/bin/remage-cpp"

echo "Local Remage Momentum environment configured."
echo "Remage wrapper: $(command -v remage)"
echo "C++ executable: $REMAGE_CPP_PATH"
echo
remage --version
