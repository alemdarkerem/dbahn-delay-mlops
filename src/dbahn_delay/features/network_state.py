"""Live network-state features: what was the network doing at seal time?

Shocks (medical emergencies, signal failures) are unpredictable, but their
knock-on delays are not — once a disruption is observed, the elevated risk
of the trains behind it is knowable. These features give the model that
signal: delay statistics of the trailing 60 minutes as they were observable
at the moment the prediction was sealed.

Leak safety (the hard part):
- Production seals a stop's prediction at the ``(scheduled_hour - 2):05``
  fetch cycle (a stop is first seen in its h+2 plan slice). Training
  replays exactly that clock: ``seal_time = scheduled_time truncated to
  the hour - 1h55m``.
- An observation enters the stream at its REALIZED event time (the ``time``
  column), i.e. a delay counts as known only once it has actually happened —
  strictly more conservative than DB's live forecasts.

Mechanics: observations are bucketed into 5-minute slots whose stats become
available at the bucket END; per scope we keep cumulative sums and recover
the trailing hour as ``cum(seal) - cum(seal - 60m)`` via two asof joins.
Seals always land on :05 — a bucket boundary — so the recovered window is
exactly ``[seal - 60m, seal)``.

``DBAHN_NETSTATE_LEAD_MIN`` replaces the seal clock with a fixed lead in
minutes before the scheduled time (diagnostic only — "what would a 30-min
seal buy?"); use multiples of 5 to keep the window exact. With leads under
60 a very early train could see its own departure — acceptable for a
diagnostic, never for production.
"""

import os

import polars as pl

BUCKET_MINUTES = 5
WINDOW_MINUTES = 60

# Scopes, most specific first. Only the station scope exports its count —
# the coarser scopes are almost never empty, so their counts carry no signal.
SCOPES: dict[str, list[str]] = {
    "station_live": ["station_name"],
    "type_live": ["train_type"],
    "network_live": [],
}

NETWORK_STATE_COLUMNS = [
    "station_live_mean_delay_60m",
    "station_live_delayed_share_60m",
    "station_live_n_60m",
    "type_live_mean_delay_60m",
    "type_live_delayed_share_60m",
    "network_live_mean_delay_60m",
    "network_live_delayed_share_60m",
]


def seal_time_expr() -> pl.Expr:
    """When production would have sealed this stop's prediction."""
    lead_min = os.environ.get("DBAHN_NETSTATE_LEAD_MIN")
    if lead_min:
        return pl.col("scheduled_time") - pl.duration(minutes=int(lead_min))
    # Fetch cycle runs at :05 and first sees a stop in the h+2 slice -> (H-2):05.
    return pl.col("scheduled_time").dt.truncate("1h") - pl.duration(hours=1, minutes=55)


def cumulative_buckets(obs: pl.LazyFrame, keys: list[str], threshold: int) -> pl.LazyFrame:
    """5-minute observation buckets -> cumulative sums keyed by availability.

    ``_avail`` is the bucket END: stats of observations in [t, t+5m) exist
    only once the bucket has closed.
    """
    bucketed = (
        obs.sort([*keys, "obs_time"])
        .group_by_dynamic("obs_time", every=f"{BUCKET_MINUTES}m", group_by=keys)
        .agg(
            _delay=pl.col("delay_in_min").sum(),
            _delayed=(pl.col("delay_in_min") >= threshold).sum(),
            _n=pl.len().cast(pl.Int64),
        )
    )
    cums = [pl.col(c).cum_sum() for c in ("_delay", "_delayed", "_n")]
    if keys:
        cums = [c.over(keys) for c in cums]
    return (
        bucketed.with_columns(*cums)
        .with_columns(_avail=pl.col("obs_time") + pl.duration(minutes=BUCKET_MINUTES))
        .select([*keys, "_avail", "_delay", "_delayed", "_n"])
        .sort("_avail")
    )


def add_network_state(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Attach the 60-minute network state as of each stop's seal time."""
    # Import here: build.py imports this module, so the top level must not
    # import build back.
    from dbahn_delay.features.build import DELAYED_THRESHOLD_MIN

    obs = lf.filter(pl.col("is_canceled").not_()).select(
        "station_name",
        "train_type",
        obs_time=pl.col("time"),
        delay_in_min=pl.col("delay_in_min"),
    )
    out = (
        lf.with_columns(_seal=seal_time_expr())
        .with_columns(_seal_lo=pl.col("_seal") - pl.duration(minutes=WINDOW_MINUTES))
        .sort("_seal")  # asof joins preserve left order, so one sort serves all
    )
    for prefix, keys in SCOPES.items():
        cum = cumulative_buckets(obs, keys, DELAYED_THRESHOLD_MIN)
        hi = cum.rename({"_delay": "_hi_delay", "_delayed": "_hi_delayed", "_n": "_hi_n"})
        lo = cum.rename({"_delay": "_lo_delay", "_delayed": "_lo_delayed", "_n": "_lo_n"})
        out = (
            out.join_asof(
                hi,
                left_on="_seal",
                right_on="_avail",
                by=keys or None,
                strategy="backward",
                check_sortedness=False,
            )
            .drop("_avail", strict=False)
            .join_asof(
                lo,
                left_on="_seal_lo",
                right_on="_avail",
                by=keys or None,
                strategy="backward",
                check_sortedness=False,
            )
            .drop("_avail", strict=False)
        )
        # cum(seal) - cum(seal-60m); a missing cum row simply means "no
        # observations up to that point", i.e. zero.
        n = pl.col("_hi_n").fill_null(0) - pl.col("_lo_n").fill_null(0)
        delay = pl.col("_hi_delay").fill_null(0) - pl.col("_lo_delay").fill_null(0)
        delayed = pl.col("_hi_delayed").fill_null(0) - pl.col("_lo_delayed").fill_null(0)
        exprs = [
            pl.when(n > 0).then(delay / n).alias(f"{prefix}_mean_delay_60m"),
            pl.when(n > 0).then(delayed / n).alias(f"{prefix}_delayed_share_60m"),
        ]
        if prefix == "station_live":
            exprs.append(n.alias(f"{prefix}_n_60m"))
        out = out.with_columns(exprs).drop(
            "_hi_delay", "_hi_delayed", "_hi_n", "_lo_delay", "_lo_delayed", "_lo_n"
        )
    return out.drop("_seal", "_seal_lo")
