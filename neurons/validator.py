# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 PROTEUS Compute

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import time

import bittensor as bt

from proteus.base.validator import BaseValidatorNeuron
from proteus.validator.forward import forward as proteus_forward


class Validator(BaseValidatorNeuron):
    """PROTEUS router neuron (validator).

    Orchestrates the forward() loop (route -> query -> score), maintains the
    scores, and lets the base class push the weights on-chain on its own
    cadence (set_weights + Yuma consensus).
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("PROTEUS router initialized.")

        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """One iteration: delegates to proteus.validator.forward."""
        return await proteus_forward(self)


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            # The work runs in a background thread; this loop only reports liveness.
            # If that thread is gone, exit non-zero so the supervisor restarts the
            # process instead of leaving a container that looks healthy and does
            # nothing.
            if validator.thread is None or not validator.thread.is_alive():
                bt.logging.error(
                    "Validation thread is dead. Exiting so the supervisor restarts us."
                )
                sys.exit(1)

            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
