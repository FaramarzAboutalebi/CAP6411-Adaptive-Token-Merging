Weights: random (timing only)

## Batch size 1

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res208 | reduced resolution | 169 | 4.28 | 4.32 | 0.93× | 384 |
| res160 | reduced resolution | 100 | 4.28 | 4.32 | 0.93× | 385 |
| res224 | reduced resolution | 196 | 4.27 | 4.31 | 0.93× | 384 |
| res144 | reduced resolution | 81 | 4.24 | 4.29 | 0.94× | 385 |
| res128 | reduced resolution | 64 | 4.24 | 4.27 | 0.94× | 385 |
| res176 | reduced resolution | 121 | 4.11 | 4.15 | 0.97× | 384 |
| res192 | reduced resolution | 144 | 4.09 | 4.12 | 0.97× | 384 |
| res240 | baseline | 225 | 3.98 | 4.01 | 1.00× | 384 |

## Batch size 8

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res240 | baseline | 225 | 5.96 | 5.97 | 1.00× | 679 |
| res224 | reduced resolution | 196 | 5.54 | 5.55 | 1.08× | 679 |
| res208 | reduced resolution | 169 | 5.37 | 5.40 | 1.11× | 680 |
| res160 | reduced resolution | 100 | 5.27 | 5.31 | 1.13× | 680 |
| res144 | reduced resolution | 81 | 5.25 | 5.29 | 1.14× | 680 |
| res128 | reduced resolution | 64 | 5.21 | 5.24 | 1.14× | 680 |
| res176 | reduced resolution | 121 | 5.11 | 5.14 | 1.17× | 680 |
| res192 | reduced resolution | 144 | 5.09 | 5.11 | 1.17× | 680 |

## Batch size 32

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res240 | baseline | 225 | 20.33 | 20.33 | 1.00× | 1694 |
| res224 | reduced resolution | 196 | 18.73 | 18.74 | 1.09× | 1693 |
| res208 | reduced resolution | 169 | 17.73 | 17.74 | 1.15× | 1694 |
| res192 | reduced resolution | 144 | 16.31 | 16.31 | 1.25× | 1694 |
| res176 | reduced resolution | 121 | 15.04 | 15.05 | 1.35× | 1694 |
| res160 | reduced resolution | 100 | 13.68 | 13.68 | 1.49× | 1694 |
| res144 | reduced resolution | 81 | 13.37 | 13.37 | 1.52× | 1694 |
| res128 | reduced resolution | 64 | 12.18 | 12.19 | 1.67× | 1694 |

## Batch size 128

| Variant | Family | Tokens (theor.) | Median ms | p90 ms | Speedup | Peak MB |
|---|---|---|---|---|---|---|
| res240 | baseline | 225 | 96.38 | 96.48 | 1.00× | 5742 |
| res224 | reduced resolution | 196 | 89.07 | 89.10 | 1.08× | 5742 |
| res208 | reduced resolution | 169 | 80.09 | 80.12 | 1.20× | 5742 |
| res192 | reduced resolution | 144 | 71.84 | 71.87 | 1.34× | 5742 |
| res176 | reduced resolution | 121 | 64.10 | 64.13 | 1.50× | 5742 |
| res160 | reduced resolution | 100 | 58.11 | 58.14 | 1.66× | 5742 |
| res144 | reduced resolution | 81 | 53.61 | 53.64 | 1.80× | 5742 |
| res128 | reduced resolution | 64 | 48.86 | 48.88 | 1.97× | 5743 |
