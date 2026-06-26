# 1 Billion/Trillion Row Challenge using Arkouda

How performant are different programming languages and libraries with data processing? The [1 billion row challenge](https://1brc.dev/) has become a popularized, rather fun way to test this. The premise is simple: Calculate the min, mean, and max of 1 billion measurements, grouped by station type. The [1 trillion row challenge](https://github.com/coiled/1trc) takes this a step further, extending the challenge to 1 trillion rows of data.

At this scale, parallel and/or distributed computing is almost certainly needed for reasonable compute times. With Chapel being a language specifically designed for parallel computing and [Arkouda](https://github.com/Bears-R-Us/arkouda) acting as a bridge between Python and parallelized Chapel code, I wanted to test how Arkouda fares against more traditional parallelized Python approaches, such as using Dask.

With that, let's start setting up Arkouda!

## Installing Anaconda, Arkouda, and Chapel
Installation was mostly straightforward (following the [documentation](https://bears-r-us.github.io/arkouda/setup/BUILD.html)), except for a few issues:
* Installing Anaconda via Homebrew doesn't automatically give access to the `conda` command! Turns out I needed to add Anaconda to PATH manually.
* I first tried to install Chapel via Homebrew, but interestingly, that didn't come with a `util/` directory to set environment variables! So I ended up building Chapel from source.

## Building the Arkouda Server
Following the steps outlined in the [documentation](https://bears-r-us.github.io/arkouda/setup/BUILD.html), I eventually ran into one test failing upon building the server:

![test_export_hdf fail](<images/test_export_hdf-fail.png>)

I describe the error in detail in [this GitHub issue](https://github.com/Bears-R-Us/arkouda/issues/5494). Essentially, a test named `test_export_hdf` fails due to mismatched columns, and this is caused by a higher-than-supported `hdf5` version `2.10.0`. I ended up resolving this by changing Arkouda's `hdf5>=1.12.2` dependency to `hdf5==1.14.6`.

Smooth sailing for now. Now after running `make`, we've got a server ready to run with `./arkouda_server`!

## Arkouda Warm-Up: Ungrouped Min, Mean, and Max
Now that I've got the Arkouda server up and running, it's time to test it out! I started with 1,000 rows of data spread across 2 Parquet files, and I used Arkouda to simply calculate the min, mean, and max of them. No grouping by stations yet, just to get a feel of what we're working with.
```py
from math import ceil

import arkouda as ak
from os.path import abspath

n = 1_000
chunksize = 500

ak.connect()

data = ak.read(
    [abspath(f"measurements-{i}.parquet") for i in range(ceil(n / chunksize))]
)
print(
    ak.mink(data["measure"], 1)[0],
    ak.mean(data["measure"]),
    ak.maxk(data["measure"], 1)[0]
)
```

Let's now do it grouped by station. My reference Dask implementation adapted from Coiled's Dask solution is as follows:

```py
import dask.dataframe as dd

df = dd.read_parquet(
    "./",
    dtype_backend="pyarrow",
)

df = df.groupby("station").agg(["min", "max", "mean"])
df = df.sort_values("station").compute()

print(df)
```

Now, let's do it with Arkouda. The built in Arkouda `grouped.min`, `grouped.mean`, and `grouped.max` functions compute the min, mean, and max across each group over three separate parallelized passes through the data.

```py
def compute_arkouda_stats(data):
    stations = data["station"]
    measures = data["measure"]

    # Sort by station name on the server so group keys are in station order.
    order = ak.argsort(stations)
    stations = stations[order]
    measures = measures[order]

    grouped = ak.GroupBy(stations, assume_sorted=True)

    # Three passes over grouped values.
    station_keys, mins = grouped.min(measures, skipna=True)
    _, means = grouped.mean(measures, skipna=True)
    _, maxs = grouped.max(measures, skipna=True)

    return station_keys, mins, means, maxs
```

This works great, but what if we wanted to calculate the min, mean, and max in only one pass? Arkouda doesn't currently provide a built-in function for that, so I took this as an opportunity to make my own custom function!

## Adding a Grouped `min_mean_max` via Segmented Reduction
You can check out the changes I've made (including Chapel code within Arkouda) [here](https://github.com/eric-vo/arkouda/commits/grouped-stats/):
* My new `GroupBy.min_mean_max(values, skipna=True)` returns the unique keys plus a min, mean, and max per group, all computed in a single server-side pass.
* On the Python client side (`groupbyclass.py`), I added a new `min_mean_max` reduction type that sends the existing `segmentedReduction` command with `op="min_mean_max"`. The server replies with three symbol names joined by `+`, which I parse back into three separate pdarrays (mins, means, maxs).
* The heavy lifting happens in Chapel (`ReductionMsg.chpl`) in a new `segMinMeanMax` proc. Instead of three reductions, it does one segmented parallel scan over the values.
* Each element is mapped to a tuple `(resetAtSegmentStart, hasValid, min, sum, max, count)`. The first element of every segment is flagged as a reset point.
* I wrote a custom scan operator, `ResettingMinMeanMaxScanOp`, that accumulates min, sum, max, and count, but resets its running state whenever it hits a segment boundary. After the scan, the last element of each segment holds the fully combined stats for that group.

After wiring that in place, we can now use our new `min_mean_max` function in our calculation!

```py
def compute_arkouda_stats(data):
    stations = data["station"]
    measures = data["measure"]

    # Sort by station name on the server so group keys are in station order.
    order = ak.argsort(stations)
    stations = stations[order]
    measures = measures[order]

    grouped = ak.GroupBy(stations, assume_sorted=True)

    # One pass over grouped values.
    station_keys, mins, means, maxs = grouped.min_mean_max(
        measures, skipna=True
    )

    return station_keys, mins, means, maxs
```

## 1 Billion Lines
Now, let's try all three approaches out on 1 billion lines! To generate the data, I modified [Coiled's `generate_data.py`](https://github.com/coiled/1trc/blob/main/generate_data.py) script to generate files locally:

```py
# This script was adapted from Jacob Tomlinson's 1BRC submission
# https://github.com/gunnarmorling/1brc/discussions/487
from math import ceil

import numpy as np
import pandas as pd

n = 1_000_000_000  # Total number of rows of data to generate
chunksize = 100_000  # Number of rows of data per file
std = 10.0  # Assume normally distributed temperatures with a standard deviation of 10
lookup_df = pd.read_csv("lookup.csv")  # Lookup table of stations and their mean temperatures


def generate_chunk(partition_idx, chunksize, std, lookup_df):
    """Generate some sample data based on the lookup table."""

    rng = np.random.default_rng(partition_idx)  # Deterministic data generation
    df = pd.DataFrame(
        {
            # Choose a random station from the lookup table for each row in our output
            "station": rng.integers(0, len(lookup_df) - 1, int(chunksize)),
            # Generate a normal distibution around zero for each row in our output
            # Because the std is the same for every station we can adjust the mean for each row afterwards
            "measure": rng.normal(0, std, int(chunksize)),
        }
    )

    # Offset each measurement by the station's mean value
    df.measure += df.station.map(lookup_df.mean_temp)
    # Round the temprature to one decimal place
    df.measure = df.measure.round(decimals=1)
    # Convert the station index to the station name
    df.station = df.station.map(lookup_df.station)

    # Save this chunk to the output file
    filename = f"measurements-{partition_idx}.parquet"
    df.to_parquet(filename, engine="pyarrow")


if __name__ == "__main__":
    for i in range(ceil(n / chunksize)):
        generate_chunk(i, chunksize, std, lookup_df)
```

Now timing each method's performance, here are the results:

![1 billion row challenge graph](images/1brc-stats.png)

A few notes:

* Initially, Dask appears to beat out Arkouda on a smaller number of nodes/locales. However, as we increase the number of available nodes, Arkouda seems to fly past Dask in terms of performance.
* Why does Dask slow down with more nodes? One possible explanation is under-parallelization, where the extra compute being added isn't necessarily parallelizing computations any further.
* It's also interesting to note that the three-pass approach seems to be superior to the one-pass approach with one node, but the one-pass approach becomes marginally faster than the 3-pass one once we start using two or more nodes. This might imply that my custom 1-pass Arkouda message is doing more than it needs to.

## 1 Trillion Lines

Generating 1 trillion lines is quite a bit heftier than generating 1 billion, so I made some tweaks to make tracking generation progress more manageable. I first modified `generate_data.py` to take in several arguments: namely, the partition ID, number of chunks to generate, chunk size, and 

## Using GitHub Copilot with Chapel