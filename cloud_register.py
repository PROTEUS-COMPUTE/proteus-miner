#!/usr/bin/env python3
"""Register this rented machine as an expert, paid to a wallet it cannot touch.

A cloud pod is someone else's computer. The usual path in the README asks a
miner to paste their 12 words so the container can rebuild their coldkey, which
is fine at home and is exactly what must never happen on rented hardware.

This takes ONE public address instead. The pod generates its own hotkey, keeps
the private half, and writes only the owner's PUBLIC key into the wallet. The
chain then records that address as the owner of the neuron, so the emission is
the owner's from the first block, and a pod that dies, is snapshotted or is
read by its host gives up nothing: there is no coldkey private key on it.

Why the extrinsic is composed here instead of calling `subtensor.register()`:
that wrapper logs `wallet.coldkey.ss58_address` before doing any work
(core/extrinsics/registration.py, "Public coldkey" line). Reading `.coldkey`
loads the PRIVATE key, which by design does not exist here, and Python builds
the f-string before it ever reaches the log level check, so the call raises no
matter how logging is configured. The five lines below are the same call the
SDK itself sends, minus that log line.

Registration must stay proof-of-work for this to hold. `burned_register` signs
with `wallet.coldkey`, the private one, so it cannot be done by a pod at all.
"""

import argparse
import os
import sys

import bittensor as bt
from bittensor.utils import is_valid_ss58_address
from bittensor.utils.registration import create_pow
from bittensor_wallet import Wallet

NETUID = int(os.getenv("NETUID", "1"))
RPC = os.getenv("SUBTENSOR_NETWORK", "wss://rpc.proteus-agent.com")
WALLET_NAME = os.getenv("WALLET_NAME", "miner")
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY", "expert1")
WALLET_PATH = os.getenv("WALLET_PATH", "~/.bittensor/wallets")


def die(msg):
    print("\n  " + msg + "\n", file=sys.stderr)
    sys.exit(2)


def owner_address() -> str:
    """The one thing the operator has to supply, checked before anything else."""
    owner = (os.getenv("PROTEUS_OWNER") or "").strip()
    if not owner:
        die(
            "PROTEUS_OWNER is not set.\n"
            "  Set it to the PRTS address that should own this miner and receive its\n"
            "  rewards. Get it from https://app.proteus-agent.com/wallet .\n"
            "  Never put your 12-word phrase on a rented machine: the address alone\n"
            "  is enough, and it is all this container can use."
        )
    if not is_valid_ss58_address(owner):
        die(
            "PROTEUS_OWNER is not a valid address:\n"
            "    %s\n"
            "  Addresses carry a checksum, so this is usually one wrong character in a\n"
            "  copy-paste. Refusing to start: mining to a mistyped address would send\n"
            "  every reward somewhere unrecoverable." % owner
        )
    return owner


def prepare_wallet(owner: str) -> Wallet:
    """Own hotkey, borrowed public coldkey. Never overwrites either."""
    w = Wallet(name=WALLET_NAME, hotkey=WALLET_HOTKEY, path=WALLET_PATH)

    base = os.path.expanduser(os.path.join(WALLET_PATH, WALLET_NAME))
    hk_file = os.path.join(base, "hotkeys", WALLET_HOTKEY)
    if not os.path.exists(hk_file):
        print("  creating this pod's own hotkey")
        w.create_new_hotkey(n_words=12, use_password=False, overwrite=True, suppress=True)
    else:
        # A restart must reuse the identity that is already registered and scored.
        print("  hotkey already present, reusing it")

    # Adopt the owner's PUBLIC key. There is no private counterpart to supply,
    # which is the whole point: this pod can be paid, it cannot spend.
    # The existence test comes first and short-circuits: reading .coldkeypub on
    # a wallet that has none raises.
    ckpub = os.path.join(base, "coldkeypub.txt")
    if not os.path.exists(ckpub) or w.coldkeypub.ss58_address != owner:
        w.regenerate_coldkeypub(ss58_address=owner, overwrite=True)

    if w.coldkeypub.ss58_address != owner:
        die("the wallet does not carry the requested owner address, refusing to continue")
    if os.path.exists(os.path.join(base, "coldkey")):
        die(
            "a coldkey PRIVATE key is present in this container.\n"
            "  That must never happen on rented hardware. Remove it and redeploy."
        )
    return w


def register(w: Wallet, sub: "bt.Subtensor") -> bool:
    """Proof of work, then the same extrinsic the SDK sends, without its log line."""
    pow_result = create_pow(
        subtensor=sub,
        wallet=w,
        netuid=NETUID,
        output_in_place=False,
        cuda=False,
        num_processes=int(os.getenv("POW_PROCESSES", "0")) or None,
        log_verbose=False,
    )
    if pow_result is None:
        return False

    call = sub.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register",
        call_params={
            "netuid": NETUID,
            "block_number": pow_result.block_number,
            "nonce": pow_result.nonce,
            "work": [int(b) for b in pow_result.seal],
            "hotkey": w.hotkey.ss58_address,
            "coldkey": w.coldkeypub.ss58_address,   # PUBLIC key, no signature needed
        },
    )
    extrinsic = sub.substrate.create_signed_extrinsic(call=call, keypair=w.hotkey)
    resp = sub.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True,
                                          wait_for_finalization=False)
    resp.process_events()
    if not resp.is_success:
        print("  registration rejected: %s" % resp.error_message)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare and check the wallet, touch no chain")
    args = ap.parse_args()

    owner = owner_address()
    print("\nPROTEUS cloud expert")
    print("  rewards owner : %s" % owner)

    w = prepare_wallet(owner)
    print("  this pod      : %s" % w.hotkey.ss58_address)
    print("  coldkey priv  : absent, as intended")

    if args.dry_run:
        print("\n  dry run, chain untouched\n")
        return 0

    print("  network       : %s" % RPC)
    sub = bt.subtensor(network=RPC)

    if sub.is_hotkey_registered(netuid=NETUID, hotkey_ss58=w.hotkey.ss58_address):
        print("  already registered on netuid %d, nothing to do\n" % NETUID)
        return 0

    print("\n  registering (proof of work, this takes a while and is normal)")
    if not register(w, sub):
        die("registration failed. The pod is running and billing, so fix or destroy it.")

    uid = sub.get_uid_for_hotkey_on_subnet(w.hotkey.ss58_address, NETUID)
    print("\n  registered, uid %s, owner %s\n" % (uid, owner))
    return 0


if __name__ == "__main__":
    sys.exit(main())
