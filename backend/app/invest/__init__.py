"""Simulated brokerage / INVEST feature package.

This package implements the paper-trading wallet experience on top of the core
GiffMeMoney backend: a thread-safe in-memory account store, a pluggable
(simulated) payment provider, the wallet / portfolio services, value history
backfill, and an allocation advisor. No real money moves and no sensitive card
data is ever persisted — see ``docs/INVEST.md`` for the full contract.
"""
